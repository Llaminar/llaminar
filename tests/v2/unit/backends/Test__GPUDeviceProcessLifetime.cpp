/**
 * @file Test__GPUDeviceProcessLifetime.cpp
 * @brief Replay actual lexical retirement against late fake runtime exit callbacks.
 *
 * Each child uses the production process owner and context pool with hardware-free
 * worker factories. Late callbacks inspect weak owners: no native driver, memory,
 * model, or alternate context registry is used. A fresh executable isolates the
 * intentionally terminal process state and the original-order negative control.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
    using namespace llaminar2;
    std::vector<std::weak_ptr<IWorkerGPUContext>> observed_contexts;
    int registered_callbacks = 0;
    int executed_callbacks = 0;
    int factory_calls = 0;
    std::string executable;

    /** @brief Fail a child without pretending runtime retirement succeeded. */
    [[noreturn]] void childFailure(const char *message, int status = 70)
    {
        std::fprintf(stderr, "GPU process lifetime replay: %s\n", message);
        std::_Exit(status);
    }
    /** @brief Stand in for a runtime exit handler registered after native setup. */
    void fakeRuntimeExit()
    {
        for (const auto &context : observed_contexts)
            if (!context.expired()) childFailure("runtime exited before exact context retirement", 42);
        ++executed_callbacks;
    }
    /** @brief Last callback authenticates that every late handler really executed. */
    void verifyCallbacks()
    {
        if (executed_callbacks != registered_callbacks)
            childFailure("late runtime callbacks were not all executed");
    }
    /** @brief Register late callbacks only in the replay, never production ownership. */
    void registerLateRuntimeExit()
    {
        if (std::atexit(fakeRuntimeExit) != 0) childFailure("cannot register fake native callback");
        ++registered_callbacks;
    }
    /** @brief Register fake factories whose actual admission uses the production gate. */
    void configureFactories(bool require_owner = true)
    {
        auto factory = [require_owner](int ordinal) -> std::unique_ptr<IWorkerGPUContext> {
            if (require_owner) GPUDeviceProcessLifetime::requireActive();
            ++factory_calls;
            return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(ordinal);
        };
        auto &pool = GPUDeviceContextPool::instance();
        pool.registerNvidiaFactory(factory, 2);
        pool.registerAMDFactory(factory, 2);
    }
    /** @brief Retain then release one exact generation before its late runtime callback. */
    void admit(DeviceId device)
    {
        auto owner = GPUDeviceContextPool::instance().retainContext(device);
        observed_contexts.push_back(owner);
        owner.reset();
        registerLateRuntimeExit();
    }
    /** @brief Require closed admission without swallowing unrelated exceptions. */
    void requireClosed()
    {
        try { GPUDeviceProcessLifetime::requireActive(); }
        catch (const std::logic_error &) { return; }
        childFailure("native admission stayed open after retirement");
    }
    /** @brief Sweep late second-device/vendor setup and reset/recreation. */
    void lateAdmissions(bool cuda_first)
    {
        GPUDeviceProcessLifetime lifetime;
        configureFactories();
        const auto first = cuda_first ? DeviceId::cuda(0) : DeviceId::rocm(0);
        const auto later_vendor = cuda_first ? DeviceId::rocm(1) : DeviceId::cuda(1);
        const auto second_device = cuda_first ? DeviceId::cuda(1) : DeviceId::rocm(1);
        admit(first);
        admit(later_vendor);
        {
            auto retired = GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(first);
            if (!retired.receipt().retiredLiveContext()) childFailure("generation reset lost its owner");
        }
        if (!observed_contexts.front().expired()) childFailure("old generation survived reset");
        admit(first);
        admit(second_device);
        // Exit this lexical scope before any of the later registered callbacks.
    }
    /** @brief Execute one isolated scenario; no child initializes a GPU runtime. */
    int runChild(const std::string &mode)
    {
        const rlimit no_core{0, 0};
        (void)setrlimit(RLIMIT_CORE, &no_core);
        alarm(10); // A broken retirement lock fails this bounded device-free replay.
        if (std::atexit(verifyCallbacks) != 0) childFailure("cannot register final callback proof");
        if (mode == "late-cuda-first") lateAdmissions(true);
        else if (mode == "late-rocm-first") lateAdmissions(false);
        else if (mode == "exception")
        {
            try
            {
                GPUDeviceProcessLifetime lifetime;
                configureFactories();
                admit(DeviceId::cuda(0));
                admit(DeviceId::rocm(1));
                throw std::runtime_error("intentional admission unwind");
            }
            catch (const std::runtime_error &error)
            {
                if (std::strcmp(error.what(), "intentional admission unwind") != 0)
                    childFailure("unexpected exception masked the unwind replay");
                requireClosed();
            }
        }
        else if (mode == "state")
        {
            configureFactories();
            GPUDeviceProcessLifetime::retireActive(); // CPU/no-owner shutdown is inert.
            if (factory_calls != 0) childFailure("no-owner shutdown probed a native factory");
            requireClosed();
            bool absent_lease_rejected = false;
            try { (void)GPUDeviceProcessLifetime::acquireNativeContext(); }
            catch (const std::logic_error &) { absent_lease_rejected = true; }
            if (!absent_lease_rejected) childFailure("native lease admitted before entry owner");
            GPUDeviceProcessLifetime lifetime;
            GPUDeviceProcessLifetime::requireActive();
            if (factory_calls != 0) childFailure("process scope construction probed a native factory");
            {
                auto lease = GPUDeviceProcessLifetime::acquireNativeContext();
                auto moved = std::move(lease);
            }
            try
            {
                auto partial = GPUDeviceProcessLifetime::acquireNativeContext();
                throw std::runtime_error("native construction unwind");
            }
            catch (const std::runtime_error &) {}

            bool nested_rejected = false;
            try { GPUDeviceProcessLifetime nested; }
            catch (const std::logic_error &) { nested_rejected = true; }
            if (!nested_rejected) childFailure("nested process owner was accepted");
            configureFactories(); admit(DeviceId::cuda(0));
            lifetime.retire(); lifetime.retire(); GPUDeviceProcessLifetime::retireActive();
            requireClosed();
            bool reopen_rejected = false;
            try { GPUDeviceProcessLifetime reopened; }
            catch (const std::logic_error &) { reopen_rejected = true; }
            if (!reopen_rejected) childFailure("terminal admission reopened");
            bool late_rejected = false;
            try { (void)GPUDeviceContextPool::instance().retainNvidiaContext(1); }
            catch (const std::logic_error &) { late_rejected = true; }
            if (!late_rejected) childFailure("late worker construction was accepted");
            bool late_lease_rejected = false;
            try { (void)GPUDeviceProcessLifetime::acquireNativeContext(); }
            catch (const std::logic_error &) { late_lease_rejected = true; }
            if (!late_lease_rejected) childFailure("native lease admitted after terminal retirement");
        }
        else if (mode == "concurrent")
        {
            GPUDeviceProcessLifetime lifetime;
            configureFactories(); admit(DeviceId::cuda(0)); admit(DeviceId::rocm(1));
            auto first = std::async(std::launch::async, [] { GPUDeviceProcessLifetime::retireActive(); });
            auto second = std::async(std::launch::async, [] { GPUDeviceProcessLifetime::retireActive(); });
            first.get(); second.get(); lifetime.retire(); requireClosed();
        }
        else if (mode == "borrowed")
        {
            GPUDeviceProcessLifetime lifetime;
            configureFactories();
            auto borrowed = GPUDeviceContextPool::instance().retainNvidiaContext(0);
            lifetime.retire(); // Must terminate before reporting successful release.
            childFailure("retirement discarded a retained context owner");
        }
        else if (mode == "direct-native")
        {
            GPUDeviceProcessLifetime lifetime;
            auto direct = GPUDeviceProcessLifetime::acquireNativeContext();
            lifetime.retire();
            childFailure("retirement discarded a direct native context");
        }
        else if (mode == "paused-native-admission")
        {
            GPUDeviceProcessLifetime lifetime;
            std::promise<void> admitted, continue_construction;
            auto permission = continue_construction.get_future();
            auto constructor = std::async(std::launch::async, [&] {
                auto native_lease = GPUDeviceProcessLifetime::acquireNativeContext();
                admitted.set_value(); permission.wait();
            });
            admitted.get_future().wait();
            lifetime.retire();
            // Exit immediately if retirement wrongly returns; do not let a later
            // lease destructor abort and masquerade as the expected pre-return failure.
            childFailure("retirement raced an admitted native constructor");
        }
        else if (mode == "original-order")
        {
            // Original code registered the pool destructor before native runtime
            // callbacks and relied on static order. No lexical retirement occurs.
            configureFactories(false);
            admit(DeviceId::cuda(0));
            return 0; // fakeRuntimeExit must detect the still-live exact owner.
        }
        else childFailure("unknown replay scenario");
        return 0;
    }
    /** @brief Launch a clean executable without sharing terminal lifecycle state. */
    int childStatus(const char *mode)
    {
        const pid_t child = fork();
        if (child < 0) throw std::runtime_error("fork failed in process lifetime replay");
        if (child == 0)
        {
            execl(executable.c_str(), executable.c_str(), "--lifetime-child", mode, nullptr);
            std::_Exit(127);
        }
        int status = 0;
        while (waitpid(child, &status, 0) < 0)
            if (errno != EINTR) throw std::runtime_error("waitpid failed in process lifetime replay");
        return status;
    }
}

/** @test Later CUDA/HIP device setup cannot reorder cleanup after runtime exit. */
TEST(GPUDeviceProcessLifetime, LateVendorDeviceAndGenerationSetup)
{
    for (const char *mode : {"late-cuda-first", "late-rocm-first"})
    {
        SCOPED_TRACE(mode); const int status = childStatus(mode);
        ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}
/** @test Exception unwind performs the same production retirement. */
TEST(GPUDeviceProcessLifetime, ExceptionUnwindRetiresBeforeRuntimeCallbacks)
{
    const int status = childStatus("exception");
    ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status), 0);
}
/** @test Admission is unique, idempotently retired and never reopened. */
TEST(GPUDeviceProcessLifetime, TerminalAdmissionState)
{
    const int status = childStatus("state");
    ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status), 0);
}
/** @test Concurrent canonical shutdown callers observe the same completion. */
TEST(GPUDeviceProcessLifetime, ConcurrentRetirementIsIdempotent)
{
    const int status = childStatus("concurrent");
    ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status), 0);
}
/** @test Outstanding native owners fail fatally before retirement can succeed. */
TEST(GPUDeviceProcessLifetime, OutstandingBorrowIsFatal)
{
    const int status = childStatus("borrowed");
    ASSERT_TRUE(WIFSIGNALED(status)); EXPECT_EQ(WTERMSIG(status), SIGABRT);
}
/** @test Direct workers and admitted-but-not-constructed workers block terminal success. */
TEST(GPUDeviceProcessLifetime, NativeAdmissionLeaseClosesConstructionRace)
{
    for (const char *mode : {"direct-native", "paused-native-admission"})
    {
        SCOPED_TRACE(mode); const int status = childStatus(mode);
        ASSERT_TRUE(WIFSIGNALED(status)); EXPECT_EQ(WTERMSIG(status), SIGABRT);
    }
}
/** @test The original static-pool ordering is detected by the same late callback. */
TEST(GPUDeviceProcessLifetime, OriginalStaticOrderNegativeControl)
{
    const int status = childStatus("original-order");
    ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status), 42);
}

/** @brief Own only replay subprocess selection; GPU admission remains entirely fake. */
int main(int argc, char **argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--lifetime-child") == 0)
        return runChild(argv[2]);
    executable = "/proc/self/exe";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
