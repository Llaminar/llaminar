/**
 * @file Test__BackendStartupConfig.cpp
 * @brief Unit regressions for the authoritative accelerator startup policy.
 *
 * These tests are deliberately device-free. They lock down parsing and prove
 * that CPU-only inventory detection returns before any vendor enumeration is
 * reachable. Real-driver primary-context isolation is covered by the matching
 * production-test preflight integration test.
 */

#include <gtest/gtest.h>

#include "backends/HardwareInventory.h"
#include "utils/DebugEnv.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

using namespace llaminar2;

namespace
{
    /**
     * @brief Restore all accelerator-startup variables after one test scope.
     */
    class ScopedBackendStartupEnvironment
    {
    public:
        ScopedBackendStartupEnvironment()
        {
            for (std::size_t index = 0; index < names_.size(); ++index)
            {
                if (const char *value = std::getenv(names_[index]))
                    saved_[index] = value;
                unsetenv(names_[index]);
            }
            mutableDebugEnv().backend_startup.reload();
        }

        ~ScopedBackendStartupEnvironment()
        {
            for (std::size_t index = 0; index < names_.size(); ++index)
            {
                if (saved_[index].has_value())
                    setenv(names_[index], saved_[index]->c_str(), 1);
                else
                    unsetenv(names_[index]);
            }
            mutableDebugEnv().backend_startup.reload();
        }

        ScopedBackendStartupEnvironment(
            const ScopedBackendStartupEnvironment &) = delete;
        ScopedBackendStartupEnvironment &operator=(
            const ScopedBackendStartupEnvironment &) = delete;

        /**
         * @brief Set one variable and refresh the global typed snapshot.
         * @param name Exact environment-variable name.
         * @param value Integer-toggle text.
         */
        void set(const char *name, const char *value)
        {
            setenv(name, value, 1);
            mutableDebugEnv().backend_startup.reload();
        }

    private:
        static constexpr std::array<const char *, 5> names_ = {
            "LLAMINAR_FORCE_CPU_ONLY_STARTUP",
            "LLAMINAR_SKIP_CUDA_STARTUP",
            "LLAMINAR_SKIP_ROCM_STARTUP",
            "HSA_USERPTR_FOR_PAGED_MEM",
            "HSA_USE_SVM",
        };
        std::array<std::optional<std::string>, names_.size()> saved_{};
    };
} // namespace

TEST(Test__BackendStartupConfig, DefaultPolicyAllowsBothAcceleratorBackends)
{
    ScopedBackendStartupEnvironment environment;
    const auto &policy = debugEnv().backend_startup;

    EXPECT_FALSE(policy.force_cpu_only);
    EXPECT_FALSE(policy.skip_cuda);
    EXPECT_FALSE(policy.skip_rocm);
    EXPECT_TRUE(policy.acceleratorsEnabled());
    EXPECT_TRUE(policy.cudaEnabled());
    EXPECT_TRUE(policy.rocmEnabled());
}

TEST(Test__BackendStartupConfig, SelectiveBackendExclusionIsSymmetric)
{
    ScopedBackendStartupEnvironment environment;

    environment.set("LLAMINAR_SKIP_CUDA_STARTUP", "1");
    EXPECT_FALSE(debugEnv().backend_startup.cudaEnabled());
    EXPECT_TRUE(debugEnv().backend_startup.rocmEnabled());

    environment.set("LLAMINAR_SKIP_CUDA_STARTUP", "0");
    environment.set("LLAMINAR_SKIP_ROCM_STARTUP", "-7");
    EXPECT_TRUE(debugEnv().backend_startup.cudaEnabled());
    EXPECT_FALSE(debugEnv().backend_startup.rocmEnabled());
}

TEST(Test__BackendStartupConfig, CpuOnlyPolicyOverridesSelectiveBackendFlags)
{
    ScopedBackendStartupEnvironment environment;
    environment.set("LLAMINAR_SKIP_CUDA_STARTUP", "0");
    environment.set("LLAMINAR_SKIP_ROCM_STARTUP", "0");
    environment.set("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "2");

    const auto &policy = debugEnv().backend_startup;
    EXPECT_TRUE(policy.force_cpu_only);
    EXPECT_FALSE(policy.acceleratorsEnabled());
    EXPECT_FALSE(policy.cudaEnabled());
    EXPECT_FALSE(policy.rocmEnabled());
}

TEST(Test__BackendStartupConfig, ParsingPreservesHistoricalIntegerSemantics)
{
    ScopedBackendStartupEnvironment environment;
    environment.set("LLAMINAR_SKIP_CUDA_STARTUP", "true");
    environment.set("LLAMINAR_SKIP_ROCM_STARTUP", "1suffix");

    // These values deliberately mirror the old std::atoi contract. Changing
    // accepted startup syntax must be an explicit CLI/configuration decision.
    EXPECT_TRUE(debugEnv().backend_startup.cudaEnabled());
    EXPECT_FALSE(debugEnv().backend_startup.rocmEnabled());
}

TEST(Test__BackendStartupConfig, VendorHostBackingRequestPreservesExactStartupIntent)
{
    ScopedBackendStartupEnvironment environment;
    EXPECT_FALSE(debugEnv().backend_startup.rocm_userptr_for_paged_mem.has_value());
    EXPECT_FALSE(debugEnv().backend_startup.rocm_use_svm.has_value());
    for (const char *value : {"0", "1", "", "00"})
    {
        environment.set("HSA_USERPTR_FOR_PAGED_MEM", value);
        EXPECT_EQ(debugEnv().backend_startup.rocm_userptr_for_paged_mem, value);
        environment.set("HSA_USE_SVM", value);
        EXPECT_EQ(debugEnv().backend_startup.rocm_use_svm, value);
    }
}

TEST(Test__BackendStartupConfig, CpuOnlyInventoryNeverPublishesGpuDevices)
{
    ScopedBackendStartupEnvironment environment;
    environment.set("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1");

    const HardwareInventory inventory = HardwareInventory::detect();
    EXPECT_FALSE(inventory.cpu_sockets.empty());
    EXPECT_TRUE(inventory.cuda_devices.empty());
    EXPECT_TRUE(inventory.rocm_devices.empty());
    EXPECT_FALSE(inventory.cuda_p2p.has_value());
    EXPECT_FALSE(inventory.rocm_p2p.has_value());
}
