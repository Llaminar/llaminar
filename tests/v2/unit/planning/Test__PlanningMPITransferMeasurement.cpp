/**
 * @file Test__PlanningMPITransferMeasurement.cpp
 * @brief Device-free payload geometry and bounded workspace accounting proofs.
 *
 * These tests enter neither MPI nor a backend. The contribution is a maximum
 * over sequential sample geometry, not a live capacity ledger or topology guess.
 */
#include "planning/PlanningMPITransferMeasurement.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

TEST(PlanningMPITransferMeasurement, WorkspaceReusesExactLargestPairWithoutSummingSamples)
{
    const std::vector<PlanningMPITransferRequest> requests{{0, 1, 17, 31}, {1, 2, 1024, 3}, {2, 0, 19, 23}};
    EXPECT_EQ(PlanningMPITransferMeasurement::workspaceBytes(requests, 0, 4), 48u);
    EXPECT_EQ(PlanningMPITransferMeasurement::workspaceBytes(requests, 1, 4), 1027u);
    EXPECT_EQ(PlanningMPITransferMeasurement::workspaceBytes(requests, 2, 4), 1027u);
    EXPECT_EQ(PlanningMPITransferMeasurement::workspaceBytes(requests, 3, 4), 0u);
    EXPECT_EQ(PlanningMPITransferMeasurement::workspaceBytes({}, 0, 1), 0u);
}

TEST(PlanningMPITransferMeasurement, RejectsInvalidGeometryEvenOnUninvolvedRanks)
{
    for (int defect = 0; defect != 10; ++defect)
    {
        PlanningMPITransferRequest request{0, 1, 17, 31};
        if (defect == 0) request.initiator = -1;
        if (defect == 1) request.responder = 3;
        if (defect == 2) request.responder = 0;
        if (defect == 3) request.request_bytes = 0;
        if (defect == 4) request.reply_bytes = 0;
        if (defect == 5) request.request_bytes = size_t(std::numeric_limits<int>::max()) + 1;
        if (defect == 6) request.reply_bytes = std::numeric_limits<size_t>::max();
        const int rank = defect == 7 ? -1 : defect == 8 ? 3 : 2;
        const int count = defect == 9 ? 0 : 3;
        EXPECT_THROW(PlanningMPITransferMeasurement::workspaceBytes({&request, 1}, rank, count), std::invalid_argument);
    }
}
