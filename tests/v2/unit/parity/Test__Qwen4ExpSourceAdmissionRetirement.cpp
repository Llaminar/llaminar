/**
 * @file Test__Qwen4ExpSourceAdmissionRetirement.cpp
 * @brief Replay failed admission observations through the real fixture sequencer.
 *
 * Captured GTest failures remain failures while callback-local aliases expire
 * before final retirement. No backend, model or device is initialized. Failed
 * cleanup cannot publish evidence, and repeated observations retain no owners.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionObservation.h"
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace llaminar2::test::parity::qwen4exp
{
/** @test Fatal assertions return from the observer but cannot skip retirement. */
TEST(Qwen4ExpSourceAdmissionRetirement, FatalObservationRetiresAfterAliasesAcrossRepeatedRuns)
{
    for (int replay = 0; replay < 4; ++replay)
    {
        auto owner = std::make_shared<int>(replay);
        const std::weak_ptr<int> observed = owner;
        bool published = false;
        int retire_calls = 0;
        ::testing::TestPartResultArray failures;
        {
            ::testing::ScopedFakeTestPartResultReporter capture(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
            observeQwen4ExpAdmissionThenRetire(
                [&] {
                    const auto local_alias = owner;
                    ASSERT_EQ(local_alias.use_count(), 2);
                    FAIL() << "intentional failed source admission";
                },
                [&] {
                    ++retire_calls;
                    ASSERT_EQ(owner.use_count(), 1);
                    owner.reset();
                    ASSERT_TRUE(observed.expired());
                    published = true;
                });
        }
        ASSERT_EQ(failures.size(), 1);
        EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
        EXPECT_NE(std::string(failures.GetTestPartResult(0).message()).find(
            "intentional failed source admission"), std::string::npos);
        EXPECT_EQ(retire_calls, 1);
        EXPECT_TRUE(published);
        EXPECT_TRUE(observed.expired());
    }
}

/** @test Exceptions retain their identity after ordered retirement, without retry. */
TEST(Qwen4ExpSourceAdmissionRetirement, ObservationExceptionRetiresBeforePropagation)
{
    auto owner = std::make_shared<int>(7);
    const std::weak_ptr<int> observed = owner;
    int retire_calls = 0;
    try
    {
        observeQwen4ExpAdmissionThenRetire(
            [&] {
                const auto alias = owner;
                throw std::runtime_error("original admission exception");
            },
            [&] {
                ++retire_calls;
                EXPECT_EQ(owner.use_count(), 1);
                owner.reset();
            });
        FAIL() << "The original observation exception was swallowed";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_STREQ(error.what(), "original admission exception");
        EXPECT_TRUE(observed.expired());
    }
    EXPECT_EQ(retire_calls, 1);
}

/** @test A retirement failure never reaches the fixture's evidence publication. */
TEST(Qwen4ExpSourceAdmissionRetirement, FailedRetirementCannotPublishSuccess)
{
    int observations = 0, retire_calls = 0;
    bool published = false;
    const auto fail_retirement = [] { throw std::runtime_error("retirement failed"); };
    EXPECT_THROW(observeQwen4ExpAdmissionThenRetire(
        [&] { ++observations; },
        [&] {
            ++retire_calls;
            fail_retirement();
            published = true;
        }), std::runtime_error);
    EXPECT_EQ(observations, 1);
    EXPECT_EQ(retire_calls, 1);
    EXPECT_FALSE(published);
}

/** @test GTest retirement assertions also stop evidence while retaining their failure. */
TEST(Qwen4ExpSourceAdmissionRetirement, FatalRetirementCannotPublishSuccess)
{
    bool published = false;
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter capture(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        observeQwen4ExpAdmissionThenRetire([] {}, [&] {
            FAIL() << "intentional retirement failure";
            published = true;
        });
    }
    ASSERT_EQ(failures.size(), 1);
    EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
    EXPECT_FALSE(published);
}

/** @test Successful observations preserve the same explicit cleanup boundary. */
TEST(Qwen4ExpSourceAdmissionRetirement, SuccessfulObservationRetiresExactlyOnce)
{
    std::string order;
    observeQwen4ExpAdmissionThenRetire([&] { order += "observe;"; },
                                    [&] { order += "retire;publish;"; });
    EXPECT_EQ(order, "observe;retire;publish;");
}
} // namespace llaminar2::test::parity::qwen4exp
