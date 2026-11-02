/** @file Test__RetainedHostServiceTLSProvider.cpp
 * @brief Actual loader/TLS provider authentication and missing/changed-source replay without passive DSO hashing.
 */
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include <gtest/gtest.h>
extern "C" void* __tls_get_addr(void*);
namespace llaminar2::test {
namespace {
/** @brief Different immutable provider observation, never a substitute resolver. */
void unrelatedProviderObservation(){}
}
TEST(RetainedHostServiceTLSProvider, ActualSelectedResolverAuthenticatesAndRevalidates){
    const auto before=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    const auto original=RetainedHostServiceReplayAccess::tlsProvider();
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::requireTLSProvider(original,reinterpret_cast<const void*>(&::__tls_get_addr)));
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),before+1);
    for(unsigned i=0;i<20;++i)RetainedHostServiceReplayAccess::requireTLSProvider(original,reinterpret_cast<const void*>(&::__tls_get_addr));
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),before+1);
}
TEST(RetainedHostServiceTLSProvider, MissingAndChangedSelectedProviderReject){
    const auto original=RetainedHostServiceReplayAccess::tlsProvider();
    EXPECT_THROW(RetainedHostServiceReplayAccess::requireTLSProvider(original,nullptr),std::runtime_error);
    EXPECT_THROW(RetainedHostServiceReplayAccess::requireTLSProvider(original,reinterpret_cast<const void*>(&unrelatedProviderObservation)),std::logic_error);
    EXPECT_THROW(RetainedHostServiceReplayAccess::requireTLSProvider({},reinterpret_cast<const void*>(&::__tls_get_addr)),std::logic_error);
}
}
