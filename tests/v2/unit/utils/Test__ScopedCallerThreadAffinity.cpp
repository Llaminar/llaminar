/**
 * @file Test__ScopedCallerThreadAffinity.cpp
 * @brief Device-free authored native-callback replay for caller CPU ownership.
 *
 * The same scope used by HardwareInventory executes against authored thread-
 * local read/write operations. No SDK or scheduler probe is performed, so a
 * one-CPU Unit allocation can still prove mutation, exception and fatal restore
 * handling. Separate actual backend isolation/admission tests qualify its use
 * around real GPU discovery; these authored masks are not native observations.
 */
#include "utils/ScopedCallerThreadAffinity.h"
#include <gtest/gtest.h>
#include <future>
#include <stdexcept>

namespace
{
    /** @brief Authored current-thread masks and deterministic operation outcomes. */
    struct AuthoredAffinity
    {
        static inline thread_local cpu_set_t mask{};
        static inline thread_local int reads=0,writes=0,read_error=0,write_error=0;
        static inline thread_local bool ignore_write=false;
        /** @return Authored read outcome for this thread, never host hardware. */
        static int read(cpu_set_t &out) noexcept
        {
            ++reads;if(read_error)return read_error;out=mask;return 0;
        }
        /** @return Authored write outcome; other thread-local states are inaccessible. */
        static int write(const cpu_set_t &value) noexcept
        {
            ++writes;if(write_error)return write_error;if(!ignore_write)mask=value;return 0;
        }
        /** @brief Start an independent authored caller on its exact two-CPU mask. */
        static void reset()
        {
            CPU_ZERO(&mask);CPU_SET(28,&mask);CPU_SET(84,&mask);
            reads=writes=read_error=write_error=0;ignore_write=false;
        }
        /** @brief Simulate a native discovery callback retargeting ONLY its caller. */
        static void nativeCallback()
        {
            CPU_ZERO(&mask);CPU_SET(0,&mask);CPU_SET(56,&mask);
        }
    };
    using ReplayScope=llaminar2::detail::ScopedCallerThreadAffinityImpl<AuthoredAffinity>;

    /** @brief Independent replay state for each authored discovery lifetime. */
    class HardwareDiscoveryCallerAffinity : public ::testing::Test
    {
    protected:
        /** @brief Reset fake operation state without querying or rebinding CPUs. */
        void SetUp() override { AuthoredAffinity::reset(); }
    };
}

TEST_F(HardwareDiscoveryCallerAffinity, RestoresExactMaskAfterNativeCallback)
{
    const auto original=AuthoredAffinity::mask;
    { const ReplayScope scope; AuthoredAffinity::nativeCallback(); }
    EXPECT_TRUE(CPU_EQUAL(&original,&AuthoredAffinity::mask));
    EXPECT_EQ(AuthoredAffinity::writes,1);
    EXPECT_EQ(AuthoredAffinity::reads,3); // Retain, inspect, authenticate restore.
}

TEST_F(HardwareDiscoveryCallerAffinity, RestoresExactMaskWhenNativeCallbackThrows)
{
    const auto original=AuthoredAffinity::mask;
    EXPECT_THROW(([] {
        const ReplayScope scope;
        AuthoredAffinity::nativeCallback();
        throw std::runtime_error("authored vendor discovery failure");
    }()),std::runtime_error);
    EXPECT_TRUE(CPU_EQUAL(&original,&AuthoredAffinity::mask));
    EXPECT_EQ(AuthoredAffinity::writes,1);
}

TEST_F(HardwareDiscoveryCallerAffinity, UnchangedCallbackDoesNotRepublishAffinity)
{
    const auto original=AuthoredAffinity::mask;
    { const ReplayScope scope; }
    EXPECT_TRUE(CPU_EQUAL(&original,&AuthoredAffinity::mask));
    EXPECT_EQ(AuthoredAffinity::writes,0);
}

TEST_F(HardwareDiscoveryCallerAffinity, CaptureFailurePrecedesNativeCallback)
{
    AuthoredAffinity::read_error=EACCES;
    int calls=0;
    EXPECT_THROW(([&] {const ReplayScope scope;++calls;}()),std::system_error);
    EXPECT_EQ(calls,0);
    EXPECT_EQ(AuthoredAffinity::writes,0);
}

TEST_F(HardwareDiscoveryCallerAffinity, EmptyCapturePrecedesNativeCallback)
{
    CPU_ZERO(&AuthoredAffinity::mask);
    int calls=0;
    EXPECT_THROW(([&] {const ReplayScope scope;++calls;}()),std::system_error);
    EXPECT_EQ(calls,0);
    EXPECT_EQ(AuthoredAffinity::writes,0);
}

TEST_F(HardwareDiscoveryCallerAffinity, ExistingOtherWorkerMaskIsUntouched)
{
    std::promise<void> ready,release;
    auto released=release.get_future();
    auto worker=std::async(std::launch::async,[&] {
        AuthoredAffinity::reset();
        CPU_ZERO(&AuthoredAffinity::mask);CPU_SET(129,&AuthoredAffinity::mask);
        const auto original=AuthoredAffinity::mask;
        ready.set_value();released.wait();
        return CPU_EQUAL(&original,&AuthoredAffinity::mask) && AuthoredAffinity::writes==0;
    });
    ready.get_future().wait();
    { const ReplayScope scope;AuthoredAffinity::nativeCallback(); }
    release.set_value();
    EXPECT_TRUE(worker.get());
}

TEST_F(HardwareDiscoveryCallerAffinity, RestoreFailureIsFatal)
{
    EXPECT_DEATH(([] {
        const ReplayScope scope;
        AuthoredAffinity::nativeCallback();AuthoredAffinity::write_error=EIO;
    }()),"cannot restore exact native discovery caller affinity");
}

TEST_F(HardwareDiscoveryCallerAffinity, ContradictoryRestoreIsFatal)
{
    EXPECT_DEATH(([] {
        const ReplayScope scope;
        AuthoredAffinity::nativeCallback();AuthoredAffinity::ignore_write=true;
    }()),"cannot restore exact native discovery caller affinity");
}
