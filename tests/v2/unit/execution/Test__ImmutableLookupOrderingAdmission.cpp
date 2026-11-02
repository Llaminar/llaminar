/**
 * @file Test__ImmutableLookupOrderingAdmission.cpp
 * @brief Device-free rejection of invalid original remote-event admission populations.
 *
 * These tests call the production legacy cohort BOM projector. No backend or
 * source worker is created, and invalid declarations fail before certification.
 */
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::qwen4exp {
TEST(ImmutableLookupOrderingAdmission, EmptyRemotePopulationHasNoAdditionalEvents)
{
    const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<24,1u<<24};
    const auto plan=PooledImmutableCompleteCohort::submissionRetirementBOM(host,{});
    EXPECT_TRUE(plan.resources().empty());
}
TEST(ImmutableLookupOrderingAdmission, RejectsMalformedPopulationBeforeCertificateLookup)
{
    const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<24,1u<<24};
    const PhysicalMemoryResource gpu{0,DeviceId::cuda(0),1u<<24,1u<<24};
    const PhysicalMemoryResource foreign{1,DeviceId::cuda(1),1u<<24,1u<<24};
    EXPECT_THROW(PooledImmutableCompleteCohort::submissionRetirementBOM(gpu,{}),std::invalid_argument);
    EXPECT_THROW(PooledImmutableCompleteCohort::submissionRetirementBOM(host,std::array{host}),std::invalid_argument);
    EXPECT_THROW(PooledImmutableCompleteCohort::submissionRetirementBOM(host,std::array{gpu,gpu}),std::invalid_argument);
    EXPECT_THROW(PooledImmutableCompleteCohort::submissionRetirementBOM(host,std::array{gpu,foreign}),std::invalid_argument);
}
}
