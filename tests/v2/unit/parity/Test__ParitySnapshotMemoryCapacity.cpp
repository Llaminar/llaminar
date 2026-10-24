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

/** @test Raw query has a distinct exact source key and retains its existing FP32 payload envelope. */
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
    ASSERT_TRUE(selected.valid());EXPECT_EQ(selected.capacity.per_accelerator_bytes,ordinary.capacity.per_accelerator_bytes);
    EXPECT_EQ(selected.replaced_state_payload_bytes,0u);
    ASSERT_EQ(selected.capacity.authored_source_outputs.size(),1u);
    EXPECT_EQ(selected.capacity.authored_source_outputs.front(),raw);
}
