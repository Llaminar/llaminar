/**
 * @file Test__ParitySnapshotMemoryCapacity.cpp
 * @brief Metadata-only tests for parity graph-snapshot memory admission.
 *
 * These tests use tiny synthetic NPY files to prove that the production parity
 * fixture prices the selected checkpoint inventory without reading model-sized
 * numerical payloads. They also pin retained-row widening, collective aliases,
 * checked arithmetic, and the rule that decode checkpoints are not widened.
 */

#include "utils/ParitySnapshotMemoryCapacity.h"
#include "planning/MemoryPlanner.h"

#include <cnpy.h>
#include "utils/PrefillGraphBucketDefaults.h"
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>
#include <tuple>

using namespace llaminar2::test::parity;
using llaminar2::GraphSnapshotSourceOutput;
using llaminar2::GraphSnapshotSourceRole;

namespace
{
    /** @brief Unique temporary NPY directory removed after each test scope. */
    class TemporarySnapshotDirectory final
    {
    public:
        TemporarySnapshotDirectory()
        {
            const auto nonce = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
            path_ = std::filesystem::temp_directory_path() /
                ("llaminar-parity-snapshot-capacity-" +
                 std::to_string(nonce));
            std::filesystem::create_directories(path_);
        }

        ~TemporarySnapshotDirectory()
        {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }

        TemporarySnapshotDirectory(
            const TemporarySnapshotDirectory &) = delete;
        TemporarySnapshotDirectory &operator=(
            const TemporarySnapshotDirectory &) = delete;

        /** @return Owned directory path. */
        [[nodiscard]] const std::filesystem::path &path() const noexcept
        {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    /** @brief Save a deterministic float checkpoint with the requested shape. */
    void saveCheckpoint(
        const std::filesystem::path &path,
        const std::vector<std::size_t> &shape)
    {
        std::size_t elements = 1u;
        for (const std::size_t dimension : shape)
            elements *= dimension;
        std::vector<float> values(elements, 1.0f);
        cnpy::npy_save<float>(path.string(), values.data(), shape, "w");
    }
}

TEST(ParitySnapshotMemoryCapacity,
     PricesRetainedRowsDecodeAndCollectiveAliasesExactly)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(
        snapshots.path() / "prefill_hidden.npy", {3u, 2u});
    saveCheckpoint(
        snapshots.path() / "decode_step0_hidden.npy", {1u, 2u});
    saveCheckpoint(
        snapshots.path() / "prefill_collective_output.npy", {3u, 1u});

    const auto evidence = deriveParitySnapshotMemoryCapacity(
        snapshots.path(),
        /*reference_rows=*/3,
        /*retained_graph_rows=*/5,
        {"collective_output"});

    ASSERT_TRUE(evidence.valid());
    EXPECT_EQ(evidence.reference_file_count, 3u);
    EXPECT_EQ(evidence.unscaled_payload_bytes, 44u);
    EXPECT_EQ(evidence.collective_alias_bytes, 20u);
    EXPECT_EQ(evidence.capacity.per_accelerator_bytes, 88u)
        << "24 prefill bytes widen to 40, 8 decode bytes stay fixed, and "
           "12 collective bytes widen to two independently live 20-byte values";

    const auto cached = deriveParitySnapshotMemoryCapacity(
        snapshots.path(), 3, 5, {"collective_output"});
    EXPECT_EQ(cached.capacity, evidence.capacity);
    EXPECT_EQ(cached.reference_file_count, evidence.reference_file_count);
}

TEST(ParitySnapshotMemoryCapacity, RejectsInvalidAndOverflowingGeometry)
{
    TemporarySnapshotDirectory empty;
    EXPECT_THROW(
        (void)deriveParitySnapshotMemoryCapacity(
            empty.path(), 1, 1, {}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)detail::widenParityPrefillPayload(
            std::numeric_limits<std::size_t>::max(), 2, 3),
        std::overflow_error);
}

/** @test Authenticated live pool files are excluded exactly once before native state pricing. */
TEST(ParitySnapshotMemoryCapacity, ReplacesOnlySelectedAuthenticatedNativeStatePrefixes)
{
    TemporarySnapshotDirectory snapshots;
    for(const auto &[prefix,rows,pools]:std::vector<std::tuple<std::string,std::size_t,std::size_t>>{
        {"",9,2},{"decode_step0_",1,2},{"decode_step1_",1,2},{"decode_step2_",1,3}}) {
        saveCheckpoint(snapshots.path()/(prefix+"layer3_QSA_INDEXER_QUERY.npy"),{1,rows,512});
        saveCheckpoint(snapshots.path()/(prefix+"layer3_QSA_INDEXER_POOLED_KEYS.npy"),{1,pools,128});
        saveCheckpoint(snapshots.path()/(prefix+"layer3_QSA_SELECTED_INDICES.npy"),{1,rows,2051});
    }
    const auto selected=std::vector{
        GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseQuery),
        GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparsePooledKeys),
        GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseSelectedIndices)};
    const auto ordinary=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{});
    const auto native=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{},selected);
    ASSERT_TRUE(native.valid());
    EXPECT_EQ(native.replaced_state_payload_bytes,4608u);
    EXPECT_EQ(native.capacity.per_accelerator_bytes+4608u,ordinary.capacity.per_accelerator_bytes);
    auto sorted=selected;std::sort(sorted.begin(),sorted.end(),[](const auto &a,const auto &b){return a.semanticKey()<b.semanticKey();});
    EXPECT_EQ(native.capacity.authored_source_outputs,sorted);
    // Default/non-QFN policy preserves the original projection byte-for-byte.
    EXPECT_EQ(deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{}).capacity,ordinary.capacity);
    auto duplicate=selected;duplicate.push_back(selected.front());
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{},duplicate),std::invalid_argument);
    auto missing=selected;missing.push_back(GraphSnapshotSourceOutput::qwenSparse(7,GraphSnapshotSourceRole::QwenSparsePooledKeys));
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{},missing),std::invalid_argument);
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{"QSA_INDEXER_POOLED_KEYS"},selected),std::invalid_argument);
}

/** @test A source identity cannot reprice a wrong-width/unnamed oracle file as valid native state. */
TEST(ParitySnapshotMemoryCapacity, RejectsUnauthenticatedOrMalformedNativeStatePrefixes)
{
    const auto selected=std::vector{
        GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparsePooledKeys)};
    TemporarySnapshotDirectory wrong_shape;
    saveCheckpoint(wrong_shape.path()/"layer3_QSA_INDEXER_POOLED_KEYS.npy",{1,2,127});
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(wrong_shape.path(),9,9,{},selected),std::invalid_argument);
    TemporarySnapshotDirectory wrong_name;
    saveCheckpoint(wrong_name.path()/"layer03_QSA_INDEXER_POOLED_KEYS.npy",{1,2,128});
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(wrong_name.path(),9,9,{},selected),std::invalid_argument);
    TemporarySnapshotDirectory wrong_scalar;
    const std::array<std::uint32_t,256> integer_pool{};
    cnpy::npy_save<std::uint32_t>((wrong_scalar.path()/"layer3_QSA_INDEXER_POOLED_KEYS.npy").string(),
        integer_pool.data(), {1,2,128}, "w");
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(wrong_scalar.path(),9,9,{},selected),std::invalid_argument);
    TemporarySnapshotDirectory truncated;
    const auto truncated_path=truncated.path()/"layer3_QSA_INDEXER_POOLED_KEYS.npy";
    saveCheckpoint(truncated_path,{1,2,128});
    std::filesystem::resize_file(truncated_path,std::filesystem::file_size(truncated_path)-4);
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(truncated.path(),9,9,{},selected),std::invalid_argument);
    for(const auto role:{GraphSnapshotSourceRole::QwenSparseRawQuery,GraphSnapshotSourceRole::QwenSparseQuery,GraphSnapshotSourceRole::QwenSparseSelectedIndices})
    {
        const auto item=GraphSnapshotSourceOutput::qwenSparse(3,role);
        const auto columns=role==GraphSnapshotSourceRole::QwenSparseSelectedIndices ? 2051u : 512u;
        for(const auto shape:std::vector<std::vector<std::size_t>>{{1,9,columns-1},{1,8,columns},{1,0,columns},{2,9,columns}})
        {
            TemporarySnapshotDirectory malformed;
            saveCheckpoint(malformed.path()/(item.semanticKey()+".npy"),shape);
            EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(malformed.path(),9,9,{}, {item}),std::invalid_argument);
        }
        TemporarySnapshotDirectory wrong_source_scalar;
        std::vector<double> f64(9*columns);
        cnpy::npy_save<double>((wrong_source_scalar.path()/(item.semanticKey()+".npy")).string(),
            f64.data(),{1,9,columns},"w");
        EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(wrong_source_scalar.path(),9,9,{}, {item}),std::invalid_argument);
        TemporarySnapshotDirectory bad_decode_rows;
        saveCheckpoint(bad_decode_rows.path()/("decode_step0_"+item.semanticKey()+".npy"),{1,9,columns});
        EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(bad_decode_rows.path(),9,9,{}, {item}),std::invalid_argument);
    }
    TemporarySnapshotDirectory headed_query;
    saveCheckpoint(headed_query.path()/"layer3_QSA_INDEXER_QUERY.npy",{1,9,4,128});
    EXPECT_TRUE(deriveParitySnapshotMemoryCapacity(headed_query.path(),9,9,{},
        {GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseQuery)}).valid());
    TemporarySnapshotDirectory exact_integer_selection;
    std::vector<std::int64_t> integers(9*2051);
    cnpy::npy_save<std::int64_t>((exact_integer_selection.path()/"layer3_QSA_SELECTED_INDICES.npy").string(),
        integers.data(),{1,9,2051},"w");
    EXPECT_TRUE(deriveParitySnapshotMemoryCapacity(exact_integer_selection.path(),9,9,{},
        {GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseSelectedIndices)}).valid());
    TemporarySnapshotDirectory state_only;
    saveCheckpoint(state_only.path()/"layer3_QSA_INDEXER_POOLED_KEYS.npy",{1,2,128});
    const auto state=deriveParitySnapshotMemoryCapacity(state_only.path(),9,9,{},selected);
    EXPECT_TRUE(state.valid());EXPECT_EQ(state.capacity.per_accelerator_bytes,0u);
    EXPECT_EQ(state.replaced_state_payload_bytes,1024u);
}

/** @test Scalar observation count never substitutes for physical retained owner identity. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesUseMaximumScalarAndIndependentOwners)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"hidden.npy",{9u,2u});
    saveCheckpoint(snapshots.path()/"decode_step0_hidden.npy",{1u,2u});
    saveCheckpoint(snapshots.path()/"decode_step1_hidden.npy",{1u,3u});
    saveCheckpoint(snapshots.path()/"decode_step2_hidden.npy",{1u,2u});
    const auto original=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{});
    EXPECT_FALSE(original.capacity.main_phase_payload);
    EXPECT_EQ(original.capacity.per_accelerator_bytes,100u);
    const auto phases=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    ASSERT_TRUE(phases.capacity.main_phase_payload);
    const auto &payload=*phases.capacity.main_phase_payload;
    EXPECT_EQ(payload.wide_main_bytes,108u);EXPECT_EQ(payload.scalar_main_bytes,12u);
    EXPECT_EQ(payload.bytesFor(llaminar2::ServingSnapshotArenaInventory::ordinary(true,false,0,0)),132u);
    EXPECT_EQ(payload.bytesFor(llaminar2::ServingSnapshotArenaInventory::ordinary(true,true,6,1)),144u);
}
/** @test A phase contract cannot silently manufacture a missing scalar or wide producer surface. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesRejectIncompleteObservationSurface)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"hidden.npy",{9u,2u});
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases),std::invalid_argument);
}

/** @test Disjoint scalar observations price the frozen producer union, not the largest observation alone. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesPreserveDisjointScalarSources)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"A.npy",{3u,2u});
    saveCheckpoint(snapshots.path()/"decode_step0_A.npy",{1u,2u});
    saveCheckpoint(snapshots.path()/"decode_step1_B.npy",{1u,5u});
    const auto evidence=deriveParitySnapshotMemoryCapacity(snapshots.path(),3,3,{}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    ASSERT_TRUE(evidence.capacity.main_phase_payload);
    EXPECT_EQ(evidence.capacity.main_phase_payload->scalar_main_bytes,28u);
    EXPECT_EQ(evidence.capacity.main_phase_payload->wide_main_bytes,84u);
}

/** @test Prefill-only sources remain in the scalar capture union as well. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesPreservePrefillOnlySources)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"A.npy",{3u,2u});
    saveCheckpoint(snapshots.path()/"B.npy",{3u,5u});
    saveCheckpoint(snapshots.path()/"decode_step0_A.npy",{1u,2u});
    const auto evidence=deriveParitySnapshotMemoryCapacity(snapshots.path(),3,3,{}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    ASSERT_TRUE(evidence.capacity.main_phase_payload);
    EXPECT_EQ(evidence.capacity.main_phase_payload->scalar_main_bytes,28u);
    EXPECT_EQ(evidence.capacity.main_phase_payload->wide_main_bytes,84u);
}

/** @test Original serving buckets, context and MTP capacity bound the wide snapshot owner. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesIncludeOriginalPrefillBucketEnvelope)
{
    llaminar2::MTPRuntimeConfig mtp;
    for (const int cap : {512, 1024, 2048})
    {
        const auto buckets = llaminar2::prefillGraphBucketSizes(cap);
        EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, buckets, true, mtp), cap);
        EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, buckets, false, mtp), 9);
    }
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 64, {128, 512}, true, mtp), 64);
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 300, {64, 128, 512}, true, mtp), 128);
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {17}, true, mtp), 17);
    // CPU execution does not retain a serving prefill ladder. Its original
    // prompt-sized envelope remains valid with model-derived context sizing.
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 0, {512}, false, mtp), 9);
    mtp.graph_capacity_draft_tokens = 15;
    mtp.max_request_batch = 8;
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {64}, true, mtp), 128);
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {512}, true, mtp), 512);
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {512}, false, mtp), 128);
    EXPECT_THROW((void)paritySnapshotServingGraphRows(9, 0, {512}, true, mtp), std::invalid_argument);
    EXPECT_THROW((void)paritySnapshotServingGraphRows(0, 4096, {512}, true, mtp), std::invalid_argument);

    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"hidden.npy", {9u, 2u});
    saveCheckpoint(snapshots.path()/"decode_step0_hidden.npy", {1u, 2u});
    const int rows = paritySnapshotServingGraphRows(9, 4096, {64, 128, 512}, true, {});
    const auto evidence = deriveParitySnapshotMemoryCapacity(snapshots.path(), 9, rows, {}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    ASSERT_TRUE(evidence.capacity.main_phase_payload);
    EXPECT_EQ(evidence.capacity.main_phase_payload->wide_main_bytes, 512u * 2u * sizeof(float));
    EXPECT_EQ(evidence.capacity.main_phase_payload->scalar_main_bytes, 2u * sizeof(float));
    EXPECT_EQ(evidence.capacity.main_phase_payload->bytesFor(
        llaminar2::ServingSnapshotArenaInventory::ordinary(true, false, 0, 0)),
        (512u + 2u) * 2u * sizeof(float));
}

/** @test Raw query keeps its identity while its native owner replaces any header payload charge. */
TEST(ParitySnapshotMemoryCapacity, RawIndexerQueryHasIndependentAuthenticatedSourceIdentity)
{
    const auto raw=GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseRawQuery);
    EXPECT_EQ(raw.semanticKey(),"layer3_QSA_INDEXER_RAW_QUERY");
    EXPECT_EQ(GraphSnapshotSourceOutput::fromQwenSparseSemanticKey(raw.semanticKey()),raw);
    EXPECT_FALSE(raw.replacesReferencePayload());
    EXPECT_NE(raw,GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseQuery));
    EXPECT_FALSE(GraphSnapshotSourceOutput::fromQwenSparseSemanticKey("layer03_QSA_INDEXER_RAW_QUERY"));
    EXPECT_FALSE(GraphSnapshotSourceOutput::fromQwenSparseSemanticKey("predictor_layer3_QSA_INDEXER_RAW_QUERY"));
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"layer3_QSA_INDEXER_RAW_QUERY.npy",{1,9,512});
    saveCheckpoint(snapshots.path()/"decode_step0_layer3_QSA_INDEXER_RAW_QUERY.npy",{1,1,512});
    const auto ordinary=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{});
    const auto selected=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,9,{}, {raw});
    ASSERT_TRUE(selected.valid());EXPECT_GT(ordinary.capacity.per_accelerator_bytes,0u);
    EXPECT_EQ(selected.capacity.per_accelerator_bytes,0u);
    EXPECT_EQ(selected.replaced_state_payload_bytes,0u);
    ASSERT_EQ(selected.capacity.authored_source_outputs.size(),1u);
    EXPECT_EQ(selected.capacity.authored_source_outputs.front(),raw);
}

/** @test Native raw geometry needs no borrowed header; other selected roles still require their own. */
TEST(ParitySnapshotMemoryCapacity, NativeRawQueryWithoutHeaderKeepsOtherHeadersMandatory)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"EMBEDDING.npy",{9,2});
    const auto raw=GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseRawQuery);
    const auto baseline=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,16,{});
    const auto selected=deriveParitySnapshotMemoryCapacity(snapshots.path(),9,16,{}, {raw});
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.capacity.per_accelerator_bytes,baseline.capacity.per_accelerator_bytes);
    ASSERT_EQ(selected.capacity.authored_source_outputs.size(),1u);
    EXPECT_EQ(selected.capacity.authored_source_outputs.front(),raw);
    const auto prepared=GraphSnapshotSourceOutput::qwenSparse(3,GraphSnapshotSourceRole::QwenSparseQuery);
    try {
        (void)deriveParitySnapshotMemoryCapacity(snapshots.path(),9,16,{}, {raw,prepared});
        FAIL()<<"A distinct prepared producer still needs its authenticated header";
    } catch(const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find(prepared.semanticKey()),std::string::npos);
    }
}

/** @test PP admission retains only resident canonical layers while preserving wide replay geometry. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesProjectUnevenPipelineLayers)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"embedding.npy", {9u, 2u});
    saveCheckpoint(snapshots.path()/"decode_step0_embedding.npy", {1u, 2u});
    for (const int layer : {0, 1, 2})
    {
        const auto key = "layer" + std::to_string(layer) + "_ATTN_OUTPUT";
        saveCheckpoint(snapshots.path()/(key+".npy"), {9u, static_cast<std::size_t>(layer+1)});
        saveCheckpoint(snapshots.path()/("decode_step0_"+key+".npy"), {1u, static_cast<std::size_t>(layer+1)});
        saveCheckpoint(snapshots.path()/("decode_step1_"+key+".npy"), {1u, static_cast<std::size_t>(layer+1)});
    }
    const auto evidence = deriveParitySnapshotMemoryCapacity(snapshots.path(), 9, 512, {}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    ASSERT_TRUE(evidence.capacity.main_phase_payload);
    const auto &payload = *evidence.capacity.main_phase_payload;
    const auto inventory = llaminar2::ServingSnapshotArenaInventory::ordinary(true, false, 0, 0);
    ASSERT_EQ(payload.layers.size(), 3u);
    EXPECT_EQ(payload.bytesFor(inventory), 514u * (2u+1u+2u+3u) * sizeof(float));
    EXPECT_EQ(payload.bytesFor(inventory, 0, 0), 514u * (2u+1u) * sizeof(float));
    EXPECT_EQ(payload.bytesFor(inventory, 1, 2), 514u * (2u+2u+3u) * sizeof(float));
    EXPECT_EQ(payload.bytesFor(inventory, 3, 3), 514u * 2u * sizeof(float));
    // The global producer remains in both stages; repeated observations never
    // manufacture extra retained arena owners or shrink the 512-row replay lane.
    EXPECT_EQ(payload.bytesFor(inventory, 0, 0)+payload.bytesFor(inventory, 1, 2),
        payload.bytesFor(inventory)+514u*2u*sizeof(float));
    EXPECT_EQ(deriveParitySnapshotMemoryCapacity(snapshots.path(), 9, 512, {}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases).capacity, evidence.capacity);
}

/** @test Malformed ownership cannot silently omit a layer or duplicate an arena charge. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesRejectInvalidLayerOwnership)
{
    const auto inventory = llaminar2::ServingSnapshotArenaInventory::ordinary(true, false, 0, 0);
    llaminar2::MainSnapshotPhasePayload payload{.layers = {{0, 512u, 1u}, {1, 1024u, 2u}}};
    EXPECT_THROW((void)payload.bytesFor(inventory, -1, 1), std::invalid_argument);
    EXPECT_THROW((void)payload.bytesFor(inventory, 2, 1), std::invalid_argument);
    payload.layers[1].layer = 0;
    EXPECT_THROW((void)payload.bytesFor(inventory), std::invalid_argument);
    payload.layers[0].layer = -1;
    EXPECT_THROW((void)payload.bytesFor(inventory), std::invalid_argument);
    payload.layers = {{1, 512u, 1u}, {0, 1024u, 2u}};
    EXPECT_THROW((void)payload.bytesFor(inventory), std::invalid_argument);
}

/** @test Actual planner BOMs project the authenticated phase inventory onto each resident PP interval. */
TEST(ParitySnapshotMemoryCapacity, RetainedServingPhasesPlannerBOMUsesResidentLayerOwnership)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"embedding.npy", {9u, 2u});
    saveCheckpoint(snapshots.path()/"decode_step0_embedding.npy", {1u, 2u});
    for (const int layer : {0, 1, 2})
    {
        const auto key = "layer" + std::to_string(layer) + "_ATTN_OUTPUT";
        saveCheckpoint(snapshots.path()/(key+".npy"), {9u, static_cast<std::size_t>(layer+1)});
        saveCheckpoint(snapshots.path()/("decode_step0_"+key+".npy"), {1u, static_cast<std::size_t>(layer+1)});
    }
    const auto evidence = deriveParitySnapshotMemoryCapacity(snapshots.path(), 9, 512, {}, {},
        ParitySnapshotPayloadProjection::RetainedMainServingPhases);
    llaminar2::ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 3;
    profile.d_model = profile.head_dim = 64;
    profile.d_ff = 128;
    profile.n_heads = profile.n_kv_heads = 1;
    profile.vocab_size = 128;
    profile.max_seq_len = 512;
    for (const auto device : {llaminar2::DeviceId::cuda(0), llaminar2::DeviceId::rocm(0)})
    {
        llaminar2::DevicePlanConfig cfg;
        cfg.device = device;
        cfg.world_rank = 0;
        cfg.device_compute_units = 60;
        cfg.device_total_bytes = cfg.device_free_bytes = 8ull << 30;
        cfg.max_seq_len = cfg.activation_seq_len = 512;
        cfg.graph_snapshot_memory = evidence.capacity;
        cfg.graph_snapshot_memory.serving_arenas =
            llaminar2::ServingSnapshotArenaInventory::ordinary(true, false, 0, 0);
        // The independent arithmetic below represents one 512-row wide arena
        // and two scalar arenas. Both partial stages retain the global source.
        for (const auto [first, last, columns] : {
            std::array{0, 0, 3}, std::array{1, 2, 7}, std::array{0, 2, 8}})
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " layers=" << first << ':' << last);
            cfg.first_layer = first;
            cfg.last_layer = last;
            const auto plan = llaminar2::MemoryPlanner::plan(profile, {cfg});
            ASSERT_EQ(plan.devices.size(), 1u);
            const auto *resource = plan.physicalPlan().find({0, device});
            ASSERT_NE(resource, nullptr);
            const auto expected = 514u * static_cast<std::size_t>(columns) * sizeof(float);
            EXPECT_EQ(resource->bytes(llaminar2::PhysicalMemoryOwner::GraphSnapshotArena), expected);
            EXPECT_EQ(plan.devices.front().graph_snapshot_bytes(), expected);
        }
        // Reject a foreign main-layer identity even when the current stage
        // would otherwise exclude it. Partial projection cannot hide bad BOMs.
        cfg.first_layer = cfg.last_layer = 0;
        ASSERT_TRUE(cfg.graph_snapshot_memory.main_phase_payload);
        cfg.graph_snapshot_memory.main_phase_payload->layers.push_back({3, 512u, 1u});
        EXPECT_THROW((void)llaminar2::MemoryPlanner::plan(profile, {cfg}), std::invalid_argument);
    }
}

/** @test The explicit immutable observation key set prices only actual selected payloads and rejects invented/duplicate keys. */
TEST(ParitySnapshotMemoryCapacity, ObservationShardCapacityMatchesExactCaptureKeys)
{
    TemporarySnapshotDirectory snapshots;
    saveCheckpoint(snapshots.path()/"hidden.npy",{2u,3u});
    saveCheckpoint(snapshots.path()/"decode_step0_hidden.npy",{1u,3u});
    saveCheckpoint(snapshots.path()/"unselected.npy",{2u,64u});
    const auto full=deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{});
    const auto selected=deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{}, {},
        ParitySnapshotPayloadProjection::ObservationEnvelope,{"hidden"});
    EXPECT_EQ(selected.reference_file_count,2u);EXPECT_EQ(selected.capacity.per_accelerator_bytes,108u);
    EXPECT_GT(full.capacity.per_accelerator_bytes,selected.capacity.per_accelerator_bytes);
    EXPECT_EQ(deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{}).capacity,full.capacity);
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{}, {},
        ParitySnapshotPayloadProjection::ObservationEnvelope,{"hidden","hidden"}),std::invalid_argument);
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{}, {},
        ParitySnapshotPayloadProjection::ObservationEnvelope,{"missing"}),std::invalid_argument);
    EXPECT_THROW((void)deriveParitySnapshotMemoryCapacity(snapshots.path(),2,8,{},
        {GraphSnapshotSourceOutput::qwenSparse(0,GraphSnapshotSourceRole::QwenSparseRawQuery)},
        ParitySnapshotPayloadProjection::ObservationEnvelope,{"hidden"}),std::invalid_argument);
}
