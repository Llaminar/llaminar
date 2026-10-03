/**
 * @file Test__RCCLDynamicLoader.cpp
 * @brief Device-free regressions for exact RCCL dependency admission.
 *
 * Invalid dependency requests must fail without searching for a different DSO.
 * Versioned passive storage evidence also fails before an invalid ABI reaches
 * native code. These tests never initialize HIP, create a communicator, or open a GPU. Real
 * communicator/reset/capture coverage remains in ProductionTestPreflight.
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/RCCLDynamicLoader.h"
#include <dlfcn.h>

extern "C" void llaminarPartialDependencyAnchor();
extern "C" unsigned llaminarPartialHostStorageCalls();

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

    TEST(RCCLDynamicLoaderPolicy, HostStorageObservationRejectsNullOrIncompleteIdentity)
    {
        RCCLHostTransportStorage receipt;
        ASSERT_TRUE(receipt.valid());
        EXPECT_EQ(rccl_dynamic::hostTransportStorage(nullptr, receipt),
                  rccl_dynamic::ncclInvalidArgument);
        // A wrong schema is rejected before interpreting any communicator or
        // calling into the native library. No fixture needs a real device.
        const auto opaque = reinterpret_cast<void*>(std::uintptr_t{1});
        receipt.abi_version = 2;
        EXPECT_EQ(rccl_dynamic::hostTransportStorage(opaque, receipt),
                  rccl_dynamic::ncclInvalidArgument);
        receipt.abi_version = 1;
        receipt.record_bytes = sizeof(receipt) - sizeof(std::uint64_t);
        EXPECT_EQ(rccl_dynamic::hostTransportStorage(opaque, receipt),
                  rccl_dynamic::ncclInvalidArgument);
    }

    TEST(RCCLDynamicLoaderPolicy, UnavailableHostStorageObservationCannotInventAnEmptyPass)
    {
        ASSERT_FALSE(rccl_dynamic::isLoaded());
        RCCLHostTransportStorage receipt;
        const auto opaque = reinterpret_cast<void*>(std::uintptr_t{1});
        EXPECT_EQ(rccl_dynamic::hostTransportStorage(opaque, receipt),
                  rccl_dynamic::ncclInvalidUsage);
        EXPECT_TRUE(receipt.valid());
        EXPECT_EQ(receipt.same_process_pinned_connections, 0u);
    }

    TEST(RCCLDynamicLoaderPolicy, PartialAdmissionWithdrawsResolvedCallbacksBeforeClosingTheDSO)
    {
        ASSERT_FALSE(rccl_dynamic::isLoaded());
        // Resolve the actual linked fixture dependency, including its installed
        // runner location. Never encode a workspace/build path into this test.
        llaminarPartialDependencyAnchor();
        Dl_info owner{};
        ASSERT_NE(dladdr(reinterpret_cast<void*>(&llaminarPartialDependencyAnchor), &owner), 0);
        ASSERT_NE(owner.dli_fname, nullptr);
        ASSERT_EQ(llaminarPartialHostStorageCalls(), 0u);
        EXPECT_FALSE(rccl_dynamic::load(owner.dli_fname));
        EXPECT_FALSE(rccl_dynamic::isLoaded());
        EXPECT_NE(std::string(rccl_dynamic::getLastError()).find("ncclGroupEnd"), std::string::npos);
        RCCLHostTransportStorage receipt;
        EXPECT_EQ(rccl_dynamic::hostTransportStorage(
                      reinterpret_cast<void*>(std::uintptr_t{1}), receipt),
                  rccl_dynamic::ncclInvalidUsage);
        EXPECT_EQ(llaminarPartialHostStorageCalls(), 0u);
        rccl_dynamic::unload();
    }
}
