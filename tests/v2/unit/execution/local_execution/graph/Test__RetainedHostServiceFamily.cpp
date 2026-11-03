/** @file Test__RetainedHostServiceFamily.cpp
 * @brief Actual TLS/provider changes reject while unrelated loader and file-metadata churn preserve resource identity.
 */
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/local_execution/graph/RetainedHostServiceClassPolicy.h"
#include <gtest/gtest.h>
#include <dlfcn.h>
#include <limits>
#include <cstdlib>
#include <exception>
#include <optional>
#include <filesystem>
#include <unistd.h>
#include <string>
#include <vector>
#include <system_error>
#include <sys/stat.h>
#include <fcntl.h>
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

namespace llaminar2::test {
namespace {
/** @param evidence Original cold declaration.
 * @return Actual rejected evidence diagnostic; empty indicates an unexpected admission. */
std::string coldEvidenceFailure(const std::shared_ptr<const RetainedHostServiceProgram::Evidence>& evidence){
    try{(void)RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);}
    catch(const std::logic_error& error){return error.what();}
    ADD_FAILURE()<<"Changed original loader evidence unexpectedly admitted";return {};
}
/** @brief Close an actual test-only module on every assertion/exception path. */
struct ColdEvidenceModule{void* handle;~ColdEvidenceModule(){if(handle&&::dlclose(handle))std::terminate();}};
/** @brief Copy the tiny TLS fixture into original process-owned temporary storage. */
class ColdEvidenceTLSFile final {
public:
    /** @brief Create private file metadata; never link or modify a shared build artifact. */
    ColdEvidenceTLSFile() {
        const auto pattern=(std::filesystem::temp_directory_path()/"llaminar-tls-metadata-XXXXXX").string();
        std::vector<char> writable(pattern.begin(),pattern.end());writable.push_back('\0');
        const auto made=::mkdtemp(writable.data());
        if(!made)throw std::system_error(errno,std::generic_category(),"TLS fixture directory");
        directory_=made;
        try{std::filesystem::copy_file(LLAMINAR_HOST_SERVICE_TLS_CONTROL,path());}
        catch(...){std::filesystem::remove_all(directory_);throw;}
    }
    /** @brief Retire only this fixture after its module handle has closed. */
    ~ColdEvidenceTLSFile(){std::error_code error;std::filesystem::remove_all(directory_,error);if(error)std::terminate();}
    /** @return Exact copied TLS image; its source build file remains untouched. */
    std::filesystem::path path()const{return directory_/"original.so";}
    /** @return Exact fixture-owned sibling for link and replacement controls. */
    std::filesystem::path sibling()const{return directory_/"sibling.so";}
private:
    std::filesystem::path directory_;
};
}

/** @test Hardlink creation/retirement and chmod cannot reprice unchanged loaded TLS storage. */
TEST(RetainedHostServiceFamily, ColdEvidenceAcceptsTLSFileMetadataOnlyChanges){
    ColdEvidenceTLSFile file;
    ColdEvidenceModule module{::dlopen(file.path().c_str(),RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    struct stat before{};ASSERT_EQ(::stat(file.path().c_str(),&before),0);
    const auto hashes=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto original=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);
    ASSERT_EQ(::link(file.path().c_str(),file.sibling().c_str()),0);
    struct stat linked{};ASSERT_EQ(::stat(file.path().c_str(),&linked),0);
    EXPECT_EQ(linked.st_ino,before.st_ino);EXPECT_EQ(linked.st_size,before.st_size);
    EXPECT_EQ(linked.st_nlink,before.st_nlink+1);
    EXPECT_EQ(linked.st_mtim.tv_sec,before.st_mtim.tv_sec);EXPECT_EQ(linked.st_mtim.tv_nsec,before.st_mtim.tv_nsec);
    EXPECT_TRUE(linked.st_ctim.tv_sec!=before.st_ctim.tv_sec || linked.st_ctim.tv_nsec!=before.st_ctim.tv_nsec);
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    const auto linkedProgram=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(original.identity(),linkedProgram.identity());
    EXPECT_EQ(original.resourceIdentity(),linkedProgram.resourceIdentity());
    EXPECT_EQ(original.allocationBytes(),linkedProgram.allocationBytes());
    ASSERT_EQ(::unlink(file.sibling().c_str()),0);
    ASSERT_EQ(::chmod(file.path().c_str(),0600),0);
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    const auto after=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(original.identity(),after.identity());
    EXPECT_EQ(original.resourceDeclaration().tlsBytes,after.resourceDeclaration().tlsBytes);
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashes);
}

/** @test A content-change timestamp still invalidates the original loaded-file declaration. */
TEST(RetainedHostServiceFamily, ColdEvidenceRejectsTLSFileModification){
    ColdEvidenceTLSFile file;
    ColdEvidenceModule module{::dlopen(file.path().c_str(),RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    struct stat before{};ASSERT_EQ(::stat(file.path().c_str(),&before),0);
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const timespec times[]{{0,UTIME_OMIT},{before.st_mtim.tv_sec+1,before.st_mtim.tv_nsec}};
    ASSERT_EQ(::utimensat(AT_FDCWD,file.path().c_str(),times,0),0);
    EXPECT_NE(coldEvidenceFailure(evidence).find("tls_before="),std::string::npos);
}

/** @test Byte-identical replacement at the same pathname and mtime is a different original image. */
TEST(RetainedHostServiceFamily, ColdEvidenceRejectsTLSFileReplacement){
    ColdEvidenceTLSFile file;
    ColdEvidenceModule module{::dlopen(file.path().c_str(),RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    struct stat before{};ASSERT_EQ(::stat(file.path().c_str(),&before),0);
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    ASSERT_TRUE(std::filesystem::copy_file(file.path(),file.sibling()));
    const timespec times[]{{0,UTIME_OMIT},before.st_mtim};
    ASSERT_EQ(::utimensat(AT_FDCWD,file.sibling().c_str(),times,0),0);
    ASSERT_EQ(::rename(file.sibling().c_str(),file.path().c_str()),0);
    struct stat after{};ASSERT_EQ(::stat(file.path().c_str(),&after),0);
    EXPECT_NE(before.st_ino,after.st_ino);EXPECT_EQ(before.st_size,after.st_size);
    EXPECT_NE(coldEvidenceFailure(evidence).find("tls_before="),std::string::npos);
}
/** @test Actual TLS loader changes expose original module geometry without reading initializer bytes. */
TEST(RetainedHostServiceFamily, ColdEvidenceIdentifiesTLSGeometry){
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    ColdEvidenceModule module{::dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    const auto reason=coldEvidenceFailure(evidence);
    EXPECT_NE(reason.find("loader_lifecycle("),std::string::npos);
    EXPECT_NE(reason.find("module_added="),std::string::npos);
    EXPECT_NE(reason.find("tls_added="),std::string::npos);
    EXPECT_NE(reason.find("memory_bytes="),std::string::npos);
    EXPECT_NE(reason.find("host_service_tls_control"),std::string::npos);
}
/** @test Unrelated no-TLS load/unload leaves the original current-resource identity valid. */
TEST(RetainedHostServiceFamily, ColdEvidenceAcceptsNonTLSLoaderChurn){
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto before=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);
    const auto attempts=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    ColdEvidenceModule module{::dlopen(LLAMINAR_HOST_SERVICE_NON_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    const auto during=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(before.identity(),during.identity());EXPECT_EQ(before.resourceDeclaration().tlsBytes,during.resourceDeclaration().tlsBytes);
    ASSERT_EQ(::dlclose(module.handle),0);module.handle=nullptr;
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    const auto after=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(before.identity(),after.identity());EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),attempts);
}
/** @test Actual policy rejection identifies only the closed key, never environment bytes. */
TEST(RetainedHostServiceFamily, ColdEvidenceRedactsRuntimePolicyValues){
    constexpr auto key="MALLOC_ARENA_MAX";const char* value=DebugEnv::envValue(key);
    struct Restore{std::optional<std::string> original;~Restore(){if(original?::setenv("MALLOC_ARENA_MAX",original->c_str(),1) : ::unsetenv("MALLOC_ARENA_MAX"))std::terminate();}} restore{value?std::optional<std::string>(value):std::nullopt};
    ASSERT_EQ(::unsetenv(key),0);const auto evidence=RetainedHostServiceProgram::originalEvidence();
    constexpr auto secret="do-not-print-original-policy-bytes-81379";ASSERT_EQ(::setenv(key,secret,1),0);
    const auto reason=coldEvidenceFailure(evidence);
    EXPECT_NE(reason.find("runtime_policy_key=MALLOC_ARENA_MAX"),std::string::npos);EXPECT_EQ(reason.find(secret),std::string::npos);
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Reused current TLS identity preserves admission; observable TLS changes retain strict rejection. */
TEST(RetainedHostServiceFamily, ColdEvidenceRequiresCurrentTLSIdentityAcrossSameFileReload){
    ColdEvidenceModule module{::dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto before=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);
    const auto hashes=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    ASSERT_EQ(::dlclose(module.handle),0);module.handle=nullptr;
    EXPECT_FALSE(coldEvidenceFailure(evidence).empty())<<"Missing original TLS module must reject";
    module.handle=::dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL);ASSERT_NE(module.handle,nullptr);
    const auto current=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    if(before.identity()==current.identity()){
        EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    }else{
        // Loader module-ID reuse is platform-dependent; an actual observable
        // file/layout/module-ID/source/provider change stays an error.
        EXPECT_FALSE(coldEvidenceFailure(evidence).empty());
    }
    EXPECT_EQ(before.resourceDeclaration().tlsBytes,current.resourceDeclaration().tlsBytes);
    EXPECT_EQ(before.resourceDeclaration().opaqueBytes,current.resourceDeclaration().opaqueBytes);
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashes);
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test A loaded, unlinked no-TLS image contributes only a diagnostic path, never required file facts. */
TEST(RetainedHostServiceFamily, ColdEvidenceAcceptsUnlinkedLoadedNonTLSImage){
    char name[]="/tmp/llaminar-non-tls-image-XXXXXX";const auto fd=::mkstemp(name);ASSERT_GE(fd,0);ASSERT_EQ(::close(fd),0);
    struct Remove{const char* path;~Remove(){::unlink(path);}} remove{name};
    ASSERT_TRUE(std::filesystem::copy_file(LLAMINAR_HOST_SERVICE_NON_TLS_CONTROL,name,std::filesystem::copy_options::overwrite_existing));
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto before=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);
    const auto hashes=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    ColdEvidenceModule module{::dlopen(name,RTLD_NOW|RTLD_LOCAL)};ASSERT_NE(module.handle,nullptr);
    ASSERT_EQ(::unlink(name),0);
    EXPECT_NO_THROW(RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    const auto after=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceProgram::originalEvidence());
    EXPECT_EQ(before.identity(),after.identity());EXPECT_EQ(before.resourceDeclaration().tlsBytes,after.resourceDeclaration().tlsBytes);
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashes);
}
}
