/**
 * @file Test__ParityPrefillSnapshotEvidence.cpp
 * @brief Device-free regressions for request-scoped segmented snapshot proof.
 *
 * Reproduce coalesced training/parity/prefix traffic without loading a model.
 * Missing or duplicate publication, wrong geometry, and telemetry reset must
 * remain failures even when earlier lifetime evidence looks healthy.
 */
#include "integration/parity/ParityPrefillSnapshotEvidence.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test::parity
{
    namespace
    {
        /** @return Production-shaped coalesced diagnostic counter. */
        PerfStatRecord publication(uint64_t chunks, uint64_t requests)
        {
            PerfStatRecord record;
            record.domain = "forward_graph";
            record.name = "prefill_chunk_snapshot_sequence_keys";
            record.phase = "prefill";
            record.device = "CUDA:0";
            record.tags = {{"chunks", std::to_string(chunks)}, {"diagnostic_only", "true"}};
            record.count = requests;
            record.value = 912.0 * static_cast<double>(requests);
            return record;
        }
    }

    TEST(ParityPrefillSnapshotEvidence, TrainingAndPrefixSeedsDoNotRelabelTheParityRequest)
    {
        auto records = std::array{publication(3, 7), publication(62, 4)};
        const auto before = ParityPrefillSnapshotEvidence::capture(records, 3);
        records[0] = publication(3, 8);
        const auto after = ParityPrefillSnapshotEvidence::capture(records, 3);
        EXPECT_NO_THROW(after.requireSingleRequestSince(before));
        // Later mandatory prefix reseeding must not mutate the retained proof.
        records[0] = publication(3, 9);
        EXPECT_NO_THROW(after.requireSingleRequestSince(before));
        const auto lifetime = ParityPrefillSnapshotEvidence::capture(records, 3);
        EXPECT_EQ(lifetime.transactions(), 13u);
        EXPECT_THROW(lifetime.requireSingleRequestSince(before), std::logic_error);
    }

    TEST(ParityPrefillSnapshotEvidence, MissingDuplicateWrongGeometryAndResetAreRejected)
    {
        const auto initial = std::array{publication(3, 7), publication(62, 4)};
        const auto before = ParityPrefillSnapshotEvidence::capture(initial, 3);
        for (const auto &records : {
                 initial,
                 std::array{publication(3, 9), publication(62, 4)},
                 std::array{publication(3, 7), publication(62, 5)},
                 std::array{publication(3, 1), publication(62, 1)}})
        {
            const auto after = ParityPrefillSnapshotEvidence::capture(records, 3);
            EXPECT_THROW(after.requireSingleRequestSince(before), std::logic_error);
        }
        const auto changed_policy = ParityPrefillSnapshotEvidence::capture(initial, 62);
        EXPECT_THROW(changed_policy.requireSingleRequestSince(before), std::logic_error);
    }

    TEST(ParityPrefillSnapshotEvidence, EveryBackendAndEveryMalformedRecordAreChecked)
    {
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
        {
            auto records = std::array{publication(3, 1)};
            records[0].device = device;
            EXPECT_EQ(ParityPrefillSnapshotEvidence::capture(records, 3).transactions(), 1u);
            for (const char *chunks : {"", "0", "-1", "3junk", "18446744073709551616"})
            {
                records[0].tags["chunks"] = chunks;
                EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(records, 3), std::logic_error);
            }
        }
        auto records = std::array{publication(3, 1)};
        records[0].tags.erase("diagnostic_only");
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(records, 3), std::logic_error);
        records[0] = publication(3, 0);
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(records, 3), std::logic_error);
        records[0] = publication(3, 1);
        records[0].value = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(records, 3), std::logic_error);
        records[0] = publication(3, 1);
        records[0].phase = "decode";
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(records, 3), std::logic_error);
    }

    TEST(ParityPrefillSnapshotEvidence, EmptyAndSplitTaggedCounterFamilies)
    {
        const auto before = ParityPrefillSnapshotEvidence::capture({}, 3);
        const auto first = std::array{publication(3, 1)};
        EXPECT_NO_THROW(ParityPrefillSnapshotEvidence::capture(first, 3).requireSingleRequestSince(before));
        const auto split = std::array{publication(3, 1), publication(3, 1)};
        EXPECT_EQ(ParityPrefillSnapshotEvidence::capture(split, 3).transactions(), 2u);
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture(split, 3).requireSingleRequestSince(before), std::logic_error);
        EXPECT_THROW(ParityPrefillSnapshotEvidence::capture({}, 0), std::logic_error);
    }
}
