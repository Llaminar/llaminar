/**
 * @file CPUJointSnapshotOwnershipContract.inc
 * @brief Root-complete checkpoint selection for the actual world2 parity case.
 *
 * This is a cold metadata contract. It does not create a runner, materialize a
 * snapshot, or certify the numerical values of a CPU fragment or collective.
 */
#include "../../integration/parity/qwen4exp/Qwen4ExpCPUJointParityCases.h"

namespace llaminar2::test::parity
{
    TEST(CPUJointSnapshotOwnershipContract, ActualWorld2RetainsCompleteRequiredInventory)
    {
        const auto original = qwen4exp::qwen4ExpProductionDefinition(
            qwen4exp::Qwen4ExpParityTopology::ROCmCPU);
        const auto &cases = qwen4exp::qwen4ExpCPUJointWorld2NumericalCases();
        ASSERT_EQ(cases.size(), 1u);
        const auto &cell = cases.front();
        const auto config = cell.toTestConfig();
        EXPECT_EQ(config.mpi_ranks, 2);
        const auto &policy = config.graph_snapshot_policy;
        EXPECT_TRUE(policy.enabled);
        EXPECT_TRUE(policy.require_snapshot_publication);
        EXPECT_EQ(policy.required_prefill_snapshot_keys,
                  original.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(policy.required_decode_snapshot_keys,
                  original.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_EQ(policy.required_prefill_snapshot_keys.size(), 294u);
        EXPECT_EQ(policy.required_decode_snapshot_keys.size(), 294u);
        EXPECT_EQ(config.thresholds.excluded_stages, original.thresholds.excluded_stages);
    }

    TEST(CPUJointSnapshotOwnershipContract, EveryRequiredRootCheckpointAvoidsHarnessReduction)
    {
        const auto config = qwen4exp::qwen4ExpCPUJointWorld2NumericalCases().front().toTestConfig();
        ASSERT_EQ(config.collective_evidence_source,
                  ParityCollectiveEvidenceSource::PostCollectiveSnapshot);
        ASSERT_TRUE(config.thresholds.allreduce_stages.empty());
        // The production root already owns completed residual/normalization
        // values. A follower remains in its worker loop; no test-side MPI sum
        // or invented same-shaped full fragment is needed for these keys.
        for (const auto *keys : {&config.graph_snapshot_policy.required_prefill_snapshot_keys,
                                 &config.graph_snapshot_policy.required_decode_snapshot_keys})
            for (const auto &key : *keys)
            {
                SCOPED_TRACE(key);
                const auto selected = selectParitySnapshot(
                    key, false, *config.collective_evidence_source);
                EXPECT_EQ(selected.key, key);
                EXPECT_EQ(selected.reduction, ParitySnapshotReduction::Direct);
                EXPECT_FALSE(selected.requires_post_collective_key);
            }
    }
}
