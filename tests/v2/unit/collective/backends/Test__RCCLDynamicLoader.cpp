/**
 * @file Test__RCCLDynamicLoader.cpp
 * @brief Device-free regressions for exact RCCL dependency admission.
 *
 * Invalid dependency requests must fail without searching for a different DSO.
 * These tests never initialize HIP, create a communicator, or open a GPU. Real
 * communicator/reset/capture coverage remains in ProductionParityPreflight.
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/RCCLDynamicLoader.h"

namespace llaminar2::test
{
    TEST(RCCLDynamicLoaderPolicy, MissingExplicitDependencyCannotSelectAnInstalledLibrary)
    {
        // Even on a host with a working RCCL installation, this exact request
        // is invalid. A successful dlopen of any alternative is a regression.
        constexpr auto missing = "/llaminar-missing-dependency/librccl-must-not-exist.so";
        ASSERT_FALSE(rccl_dynamic::isLoaded());
        EXPECT_FALSE(rccl_dynamic::load(missing));
        EXPECT_FALSE(rccl_dynamic::isLoaded());
        EXPECT_NE(std::string(rccl_dynamic::getLastError()).find(missing), std::string::npos);
        rccl_dynamic::unload();
    }

    TEST(RCCLDynamicLoaderPolicy, EmptyExplicitDependencyCannotSelectTheConfiguredDefault)
    {
        // Only nullptr requests the configured dependency. An explicit empty
        // string must not be reinterpreted as permission to pick another DSO.
        ASSERT_FALSE(rccl_dynamic::isLoaded());
        EXPECT_FALSE(rccl_dynamic::load(""));
        EXPECT_FALSE(rccl_dynamic::isLoaded());
        EXPECT_NE(std::string(rccl_dynamic::getLastError()).find("No RCCL library configured"),
                  std::string::npos);
        rccl_dynamic::unload();
    }
}
