/** @file Test__PooledDiagnosticInstantiationSelector.cpp
 * @brief Actual observer failure selector preserves native backend/ordinal ordering without SDK calls.
 */
#include "integration/backends/PooledNativeProgramObservation.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <optional>
namespace llaminar2::test {
namespace {
/** @brief Restore exact inherited diagnostic environment after one selector replay. */
struct Environment {
    const char* ordinal="LLAMINAR_POOLED_DIAGNOSTIC_FAIL_INSTANTIATE";const char* backend="LLAMINAR_POOLED_DIAGNOSTIC_FAIL_BACKEND";
    std::optional<std::string> oldOrdinal,oldBackend;
    Environment(){if(const auto* p=std::getenv(ordinal))oldOrdinal=p;if(const auto* p=std::getenv(backend))oldBackend=p;unsetenv(ordinal);unsetenv(backend);}
    ~Environment(){if(oldOrdinal)setenv(ordinal,oldOrdinal->c_str(),1);else unsetenv(ordinal);if(oldBackend)setenv(backend,oldBackend->c_str(),1);else unsetenv(backend);}
};
}
TEST(PooledDiagnosticInstantiationSelector,OriginalUnqualifiedOrdinalAndAbsentControlRemainUnchanged){
    Environment env;using pooled_program_observation::rejectInstantiation;EXPECT_FALSE(rejectInstantiation(1));
    setenv(env.ordinal,"2",1);EXPECT_FALSE(rejectInstantiation(1,"CUDA"));EXPECT_TRUE(rejectInstantiation(2,"CUDA"));EXPECT_TRUE(rejectInstantiation(2,"ROCm"));EXPECT_TRUE(rejectInstantiation(2));
}
TEST(PooledDiagnosticInstantiationSelector,SourceBackendFirstSucceedsBeforeExactRemoteBackendFirstRejects){
    Environment env;using pooled_program_observation::rejectInstantiation;setenv(env.ordinal,"1",1);
    for(const auto* remote:{"CUDA","ROCm"}){setenv(env.backend,remote,1);const auto* source=std::string(remote)=="CUDA"?"ROCm":"CUDA";
        EXPECT_FALSE(rejectInstantiation(1,source));EXPECT_TRUE(rejectInstantiation(1,remote));EXPECT_FALSE(rejectInstantiation(2,remote));}
}
TEST(PooledDiagnosticInstantiationSelector,MalformedOrAmbiguousQualifiedControlFailsBeforeAnyNativeCall){
    GTEST_FLAG_SET(death_test_style,"threadsafe");Environment env;using pooled_program_observation::rejectInstantiation;
    setenv(env.backend,"CUDA",1);EXPECT_DEATH((void)rejectInstantiation(1,"CUDA"),"");
    setenv(env.ordinal,"1",1);EXPECT_DEATH((void)rejectInstantiation(1),"");
    EXPECT_DEATH((void)rejectInstantiation(1,"OpenCL"),"");
    for(const auto* bad:{"","cuda","ROCM","CUDA:0","CUDA,ROCm"}){setenv(env.backend,bad,1);EXPECT_DEATH((void)rejectInstantiation(1,"CUDA"),"");}
    setenv(env.backend,"CUDA",1);for(const auto* bad:{"0","1x","-1","18446744073709551616"}){setenv(env.ordinal,bad,1);EXPECT_DEATH((void)rejectInstantiation(1,"ROCm"),"");}
}
}
