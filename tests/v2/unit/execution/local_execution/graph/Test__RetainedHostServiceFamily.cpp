/** @file Test__RetainedHostServiceFamily.cpp
 * @brief Real passive TLS loading preserves portable family but invalidates original live evidence.
 */
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/local_execution/graph/RetainedHostServiceClassPolicy.h"
#include <gtest/gtest.h>
#include <dlfcn.h>
#include <limits>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include "utils/DebugEnv.h"
namespace llaminar2::test {
/** @test Ordinary declaration uses loader metadata without reading provider-library payloads. */
TEST(RetainedHostServiceFamily, DeclarationDoesNotHashProviderFiles){
    const auto attempts=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto program=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO,evidence);
    const auto observed=RetainedHostServiceProgram::evidenceObservation(evidence);
    EXPECT_EQ(observed.filesHashed,0u);
    EXPECT_EQ(observed.bytesHashed,0u);
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),attempts);
    EXPECT_GT(program.allocationBytes(),program.stackBytes());
}

/** @test Current object growth changes only exact pages, without invalidating the opaque family. */
TEST(RetainedHostServiceFamily, CurrentObjectGeometryPricesExactPages){
    auto receipts=RetainedHostServiceBudgetProjection::original().classes();
    ASSERT_FALSE(receipts.empty());
    receipts.resize(1);
    const auto original=receipts.front();
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
    const auto page=original.declaration.pageBytes;
    for(const auto bytes:{std::size_t{184},std::size_t{192},page,page+1}){
        receipts.front().declaration.objectBytes=bytes;
        const auto program=RetainedHostServiceBudgetProjection::installed(receipts).program(original.role);
        EXPECT_EQ(program.resourceIdentity(),original.resourceIdentity);
        EXPECT_EQ(program.resourceDeclaration().opaqueBytes,original.declaration.opaqueBytes);
        EXPECT_EQ(program.allocationBytes(),original.declaration.stackBytes+(bytes<=page?page:2*page));
        EXPECT_NO_THROW(program.bom(cpu));
    }
}

TEST(RetainedHostServiceFamily, PassiveTLSChangesLiveEvidenceWithoutChangingCreatorClass){
    const auto before=RetainedHostServiceProgram::originalEvidence();
    const auto first=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,before);
    void* module=::dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL);
    ASSERT_NE(module,nullptr);
    struct Close final {void* handle;~Close(){if(handle)::dlclose(handle);}} close{module};
    const auto after=RetainedHostServiceProgram::originalEvidence();
    const auto second=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,after);
    EXPECT_EQ(first.resourceIdentity(),second.resourceIdentity());
    EXPECT_NE(first.identity(),second.identity());
    EXPECT_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,before),std::logic_error);
}
TEST(RetainedHostServiceFamily, CoarsePolicyGrowsWithoutEmpiricalPopulationCeilings){
    using namespace detail;
    EXPECT_EQ(plannedRetainedHostServiceOpaqueBytes(0),2u<<20);
    EXPECT_EQ(plannedRetainedHostServiceOpaqueBytes(1u<<20),2u<<20);
    EXPECT_EQ(plannedRetainedHostServiceOpaqueBytes((1u<<20)+1),4u<<20);
    EXPECT_EQ(plannedRetainedHostServiceOpaqueBytes(65u<<20),66u<<20);
    EXPECT_THROW(plannedRetainedHostServiceOpaqueBytes(std::numeric_limits<std::size_t>::max()),std::overflow_error);
    EXPECT_NO_THROW(requireRetainedHostServiceABI("2.36",8));
    EXPECT_NO_THROW(requireRetainedHostServiceABI("2.99",8));
    EXPECT_THROW(requireRetainedHostServiceABI("3.0",8),std::invalid_argument);
    EXPECT_THROW(requireRetainedHostServiceABI("2.33",8),std::invalid_argument);
    EXPECT_THROW(requireRetainedHostServiceABI("2.36",4),std::invalid_argument);
}
TEST(RetainedHostServiceFamily, RankDescriptorRejectsInventedBudgetAndLocalGeometry){
    auto original=RetainedHostServiceBudgetProjection::original();auto receipts=original.classes();ASSERT_EQ(receipts.size(),2u);
    auto invalid=receipts;--invalid[0].declaration.opaqueBytes;
    EXPECT_THROW(RetainedHostServiceBudgetProjection::installed(invalid),std::invalid_argument);
    invalid=receipts;invalid[0].declaration.objectAlignment=3;
    EXPECT_THROW(RetainedHostServiceBudgetProjection::installed(invalid),std::invalid_argument);
    auto changed=receipts;changed[0].declaration.stackBytes+=changed[0].declaration.pageBytes;
    const auto projected=RetainedHostServiceBudgetProjection::installed(changed);
    const RetainedHostServiceInstance key{0,DeviceId::cuda(0),RetainedHostServiceRole::ImmutableLookupIO};
    RetainedHostServiceDeclarations::Builder builder(0);builder.addProjected(key,projected);auto declaration=std::move(builder).freeze();
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declaration->bom(key,cpu)),0);
    EXPECT_THROW(declaration->bind(memory,std::span{&key,std::size_t{1}}),std::logic_error);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

/** @test Closed runtime environment evidence preserves empty/unset semantics and rejects changed bytes. */
TEST(RetainedHostServiceFamily, RuntimePolicyUsesDebugEnvWithoutChangingIdentitySemantics)
{
    constexpr const char* name = "MALLOC_ARENA_MAX";
    struct Restore final
    {
        const char* name;
        std::optional<std::string> original;
        /** @brief Restore the exact original presence and bytes, including an empty declaration. */
        ~Restore()
        {
            const int result = original ? ::setenv(name, original->c_str(), 1) : ::unsetenv(name);
            if (result != 0) std::terminate();
        }
    };
    const char* original_value = DebugEnv::envValue(name);
    Restore restore{name, original_value
        ? std::optional<std::string>(original_value) : std::nullopt};
    ASSERT_EQ(::unsetenv(name), 0);
    const auto absent_evidence = RetainedHostServiceProgram::originalEvidence();
    const auto absent = RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, absent_evidence);
    ASSERT_EQ(::setenv(name, "", 1), 0);
    const auto empty = RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(absent.identity(), empty.identity());
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, absent_evidence));
    ASSERT_EQ(::setenv(name, "2", 1), 0);
    EXPECT_THROW(RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, absent_evidence), std::logic_error);
    const auto nonempty_evidence = RetainedHostServiceProgram::originalEvidence();
    const auto nonempty = RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, nonempty_evidence);
    EXPECT_NE(nonempty.identity(), empty.identity());
    EXPECT_EQ(nonempty.resourceIdentity(), empty.resourceIdentity());
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, nonempty_evidence));
    ASSERT_EQ(::setenv(name, "3", 1), 0);
    EXPECT_THROW(RetainedHostServiceReplayAccess::withEvidence(
        RetainedHostServiceRole::ImmutableLookupIO, nonempty_evidence), std::logic_error);
}
} // namespace llaminar2::test
