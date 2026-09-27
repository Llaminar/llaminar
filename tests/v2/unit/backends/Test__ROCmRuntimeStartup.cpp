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

TEST(ROCmRuntimeStartup, MissingEitherPolicyRequiresAnUninitializedRuntime)
{
    using Value = std::optional<std::string_view>;
    for (const auto &userptr : std::array{Value{}, Value{"0"}})
        for (const auto &svm : std::array{Value{}, Value{"0"}})
        {
            if (userptr && svm) continue;
            EXPECT_EQ(selectROCmHostMemoryAction(userptr, svm, ROCmRuntimeState::Uninitialized),
                ROCmHostMemoryAction::InstallExplicitOwnership);
            EXPECT_EQ(selectROCmHostMemoryAction(userptr, svm, ROCmRuntimeState::Initialized),
                ROCmHostMemoryAction::RejectLatePreparation);
        }
}

TEST(ROCmRuntimeStartup, DriverOwnedBackingIsIdempotentAcrossInitialization)
{
    for (const auto state : {ROCmRuntimeState::Uninitialized, ROCmRuntimeState::Initialized})
        EXPECT_EQ(selectROCmHostMemoryAction("0", "0", state),
            ROCmHostMemoryAction::ExplicitOwnershipConfigured);
}

TEST(ROCmRuntimeStartup, EveryOtherVendorValueIsRejectedWithoutSubstitution)
{
    for (const auto value : std::array{"", "1", "false", "00", " 0", "0suffix"})
        for (const auto state : {ROCmRuntimeState::Uninitialized, ROCmRuntimeState::Initialized})
        {
            EXPECT_EQ(selectROCmHostMemoryAction(value, "0", state),
                ROCmHostMemoryAction::RejectUserPointerBacking) << value;
            EXPECT_EQ(selectROCmHostMemoryAction("0", value, state),
                ROCmHostMemoryAction::RejectSvmRegistration) << value;
            // A conflict must win even when the other half is unspecified.
            EXPECT_EQ(selectROCmHostMemoryAction(value, std::nullopt, state),
                ROCmHostMemoryAction::RejectUserPointerBacking) << value;
            EXPECT_EQ(selectROCmHostMemoryAction(std::nullopt, value, state),
                ROCmHostMemoryAction::RejectSvmRegistration) << value;
        }
}
