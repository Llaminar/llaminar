/**
 * @file Test__ROCmRuntimeStartup.cpp
 * @brief Device-free totality of the immutable ROCr host-backing startup policy.
 *
 * These tests do not call HSA, HIP or backend discovery. Real native allocation
 * and byte-correct transport are covered by the companion preflight regression.
 */
#include "backends/rocm/ROCmRuntimeStartup.h"
#include <gtest/gtest.h>
#include <array>

using namespace llaminar2;

TEST(ROCmRuntimeStartup, MissingPolicyIsInstalledOnlyBeforeRuntimeInitialization)
{
    EXPECT_EQ(selectROCmHostBackingAction(std::nullopt, ROCmRuntimeState::Uninitialized),
        ROCmHostBackingAction::InstallDriverBacking);
    EXPECT_EQ(selectROCmHostBackingAction(std::nullopt, ROCmRuntimeState::Initialized),
        ROCmHostBackingAction::RejectLatePreparation);
}

TEST(ROCmRuntimeStartup, DriverOwnedBackingIsIdempotentAcrossInitialization)
{
    for (const auto state : {ROCmRuntimeState::Uninitialized, ROCmRuntimeState::Initialized})
        EXPECT_EQ(selectROCmHostBackingAction("0", state),
            ROCmHostBackingAction::DriverBackingConfigured);
}

TEST(ROCmRuntimeStartup, EveryOtherVendorValueIsRejectedWithoutSubstitution)
{
    for (const auto value : std::array{"", "1", "false", "00", " 0", "0suffix"})
        for (const auto state : {ROCmRuntimeState::Uninitialized, ROCmRuntimeState::Initialized})
            EXPECT_EQ(selectROCmHostBackingAction(value, state),
                ROCmHostBackingAction::RejectUserPointerBacking) << value;
}
