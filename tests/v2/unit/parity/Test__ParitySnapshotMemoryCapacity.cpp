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
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

using namespace llaminar2::test::parity;

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
