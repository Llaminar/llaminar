/**
 * @file Test__ReadAheadDiagnosticSelection.cpp
 * @brief Device-free strict diagnostic parsing before original runtime admission.
 *
 * This test only changes its own process environment and local parser snapshot.
 * The actual RuntimeConfig freeze/projection replay is registered separately by
 * the source-planning owner. No model, native stream or backend is initialized.
 */
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>

namespace llaminar2 { namespace {
/** @brief Restore the caller's exact diagnostic environment at scope exit. */
class Environment final
{
public:
    /** @brief Retain the original optional value before the local test changes it. */
    Environment(){if(const char* value=std::getenv(name))original_=value;}
    /** @brief Restore absence separately from an explicitly empty value. */
    ~Environment(){if(original_)setenv(name,original_->c_str(),1);else unsetenv(name);}
    /** @brief Install one complete local value or true absence. */
    void set(const char* value){if(value)setenv(name,value,1);else unsetenv(name);}
    static constexpr const char* name="LLAMINAR_QFN_NGRAM_LOOKUP_SCHEDULE";
private:
    std::optional<std::string> original_;
};
TEST(ReadAheadDiagnosticSelection, EveryExplicitPolicyParsesWithoutAliases)
{
    EXPECT_EQ(parseImmutableLookupSchedule("consumer_scheduled"),ImmutableLookupSchedule::ConsumerScheduled);
    EXPECT_EQ(parseImmutableLookupSchedule("early_current_hosted_candidate"),ImmutableLookupSchedule::EarlyCurrentHostedCandidate);
    EXPECT_EQ(parseImmutableLookupSchedule("early_current_resident_decode_gpu_candidate"),ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate);
    for(const auto* value:{"","auto","hosted","resident","early_current_hosted","early_current_hosted_candidate "," ConsumerScheduled","EARLY_CURRENT_HOSTED_CANDIDATE"})
        EXPECT_THROW((void)parseImmutableLookupSchedule(value),std::invalid_argument);
}
TEST(ReadAheadDiagnosticSelection, ReloadPreservesExplicitSelectionAndTrueAbsence)
{
    Environment env;env.set(nullptr);ExecutionConfig selected;
    EXPECT_FALSE(selected.qfn_ngram_lookup_schedule);
    env.set("early_current_hosted_candidate");selected.reload();
    ASSERT_TRUE(selected.qfn_ngram_lookup_schedule);
    EXPECT_EQ(*selected.qfn_ngram_lookup_schedule,ImmutableLookupSchedule::EarlyCurrentHostedCandidate);
    env.set("early_current_resident_decode_gpu_candidate");selected.reload();
    EXPECT_EQ(*selected.qfn_ngram_lookup_schedule,ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate);
    env.set("consumer_scheduled");selected.reload();
    EXPECT_EQ(*selected.qfn_ngram_lookup_schedule,ImmutableLookupSchedule::ConsumerScheduled);
    env.set(nullptr);selected.reload();EXPECT_FALSE(selected.qfn_ngram_lookup_schedule);
}
TEST(ReadAheadDiagnosticSelection, UnknownOrEmptyOverrideFailsBeforeRuntimeConstruction)
{
    Environment env;env.set("consumer_scheduled");ExecutionConfig selected;
    env.set("automatic");EXPECT_THROW(selected.reload(),std::invalid_argument);
    env.set("");EXPECT_THROW(selected.reload(),std::invalid_argument);
    env.set(nullptr);EXPECT_NO_THROW(selected.reload());EXPECT_FALSE(selected.qfn_ngram_lookup_schedule);
}
}}
