/**
 * @file Test__MPIOwnedSharedBacking.cpp
 * @brief Device-free replay of actual owner-issued backing identity and retirement.
 *
 * Tests use real POSIX files under the production process owner's directory.
 * They never fabricate a root capability, initialize MPI, or query a device.
 * Explicit replacement cases preserve the replacement and prove the original
 * descriptor remains the authority through unlink, move and peer attachment.
 */
#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
using llaminar2::MPISessionOwner;
namespace {
/** @return A basename unique within this actual test process. */
std::string name() {static std::atomic<unsigned> sequence{0};return "owned-replay-"+std::to_string(::getpid())+"-"+std::to_string(++sequence);}
/** @return The actual startup-issued directory; no ambient path is adopted. */
MPISessionOwner::SharedDirectory directory()
{auto result=MPISessionOwner::sharedDirectory();if(!result)throw std::runtime_error("Replay requires actual owner wrapper");return *result;}
}
TEST(MPIOwnedSharedBacking, OriginalDescriptorAndMappingSurviveNameRetirement)
{
    auto root=directory();const auto basename=name();auto creator=root.createExclusive(basename);
    ASSERT_EQ(::ftruncate(creator.descriptor(),4096),0);creator.requireExtent(4096);
    auto peer=root.openExisting(basename,creator.identity(),4096);
    auto *mapping=static_cast<int*>(::mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,peer.descriptor(),0));
    ASSERT_NE(mapping,MAP_FAILED);mapping[0]=417;
    const auto path=creator.path();creator.unlink();creator.unlink();
    EXPECT_FALSE(std::filesystem::exists(path));int value=0;
    EXPECT_EQ(::pread(creator.descriptor(),&value,sizeof(value),0),sizeof(value));
    EXPECT_EQ(value,417);
    peer.requireExtent(4096);
    EXPECT_EQ(::munmap(mapping,4096),0);
}
TEST(MPIOwnedSharedBacking, PeerRetiredOriginalAllowsSameNameSuccessorWithoutDeletingIt)
{
    auto root=directory();const auto basename=name();auto creator=root.createExclusive(basename);
    ASSERT_EQ(::ftruncate(creator.descriptor(),128),0);
    auto peer=root.openExisting(basename,creator.identity(),128);peer.unlink();
    auto successor=root.createExclusive(basename);ASSERT_EQ(::ftruncate(successor.descriptor(),256),0);
    const auto successor_identity=successor.identity();
    EXPECT_NO_THROW(creator.unlink());
    EXPECT_TRUE(std::filesystem::exists(successor.path()));
    EXPECT_EQ(successor.identity(),successor_identity);EXPECT_NO_THROW(successor.requireExtent(256));
    creator.requireExtent(128);peer.requireExtent(128);
}
TEST(MPIOwnedSharedBacking, PeerRejectsWrongOriginalIdentityAndExtentBeforeMapping)
{
    auto root=directory();const auto basename=name();auto creator=root.createExclusive(basename);
    ASSERT_EQ(::ftruncate(creator.descriptor(),128),0);auto wrong=creator.identity();++wrong.inode;
    EXPECT_THROW((void)root.openExisting(basename,wrong,128),std::runtime_error);
    EXPECT_THROW((void)root.openExisting(basename,creator.identity(),129),std::runtime_error);
    EXPECT_THROW(creator.requireExtent(127),std::runtime_error);
    auto peer=root.openExisting(basename,creator.identity(),128);
    EXPECT_EQ(peer.identity(),creator.identity());
}
TEST(MPIOwnedSharedBacking, RejectsTraversalReservedNulAndOverlengthNames)
{
    auto root=directory();
    for(const std::string &candidate:{std::string(),std::string("."),std::string(".."),std::string(".owner"),
        std::string("../outside"),std::string("a/b"),std::string("a\0b",3),std::string(256,'x')}) {
        EXPECT_THROW((void)root.createExclusive(candidate),std::invalid_argument);
        EXPECT_THROW((void)root.openExisting(candidate,{},0),std::invalid_argument);
    }
}
TEST(MPIOwnedSharedBacking, MissingAndSymlinkAttachmentsCannotAdoptAnotherObject)
{
    auto root=directory();const auto target=name();auto original=root.createExclusive(target);
    ASSERT_EQ(::ftruncate(original.descriptor(),64),0);const auto link=name();const auto path=root.path()+"/"+link;
    EXPECT_THROW((void)root.openExisting(name(),original.identity(),64),std::system_error);
    ASSERT_EQ(::symlink(original.path().c_str(),path.c_str()),0);
    EXPECT_THROW((void)root.openExisting(link,original.identity(),64),std::system_error);
    EXPECT_TRUE(std::filesystem::is_symlink(path));
    EXPECT_TRUE(std::filesystem::exists(original.path()));
    EXPECT_EQ(::unlink(path.c_str()),0);
}
TEST(MPIOwnedSharedBacking, RegularReplacementIsNeitherAttachedNorDeletedAsOriginal)
{
    auto root=directory();const auto basename=name();auto original=root.createExclusive(basename);
    ASSERT_EQ(::ftruncate(original.descriptor(),32),0);const auto original_identity=original.identity();
    const auto renamed=original.path()+"-retained-original";
    ASSERT_EQ(::rename(original.path().c_str(),renamed.c_str()),0);
    auto replacement=root.createExclusive(basename);
    ASSERT_EQ(::ftruncate(replacement.descriptor(),32),0);
    EXPECT_NE(replacement.identity(),original_identity);
    EXPECT_THROW((void)root.openExisting(basename,original_identity,32),std::runtime_error);
    EXPECT_THROW(original.unlink(),std::runtime_error);
    EXPECT_TRUE(std::filesystem::exists(replacement.path()));replacement.requireExtent(32);
    replacement.unlink();
    ASSERT_EQ(::unlink(renamed.c_str()),0);original.unlink();
}
TEST(MPIOwnedSharedBacking, SymlinkReplacementAndLiveNeighborRemainUntouched)
{
    auto root=directory();const auto basename=name();auto original=root.createExclusive(basename);
    auto neighbor=root.createExclusive(name());
    ASSERT_EQ(::ftruncate(neighbor.descriptor(),71),0);
    const auto renamed=original.path()+"-retained-original";
    ASSERT_EQ(::rename(original.path().c_str(),renamed.c_str()),0);
    ASSERT_EQ(::symlink(neighbor.path().c_str(),original.path().c_str()),0);
    EXPECT_THROW(original.unlink(),std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_symlink(original.path()));
    neighbor.requireExtent(71);
    EXPECT_EQ(::unlink(original.path().c_str()),0);
    ASSERT_EQ(::unlink(renamed.c_str()),0);original.unlink();
}
TEST(MPIOwnedSharedBacking, MovedObjectsRejectUseAndRetainOnePhysicalDescriptor)
{
    auto source=directory();auto destination=std::move(source);
    EXPECT_THROW((void)source.identity(),std::logic_error);
    EXPECT_THROW((void)source.path(),std::logic_error);
    EXPECT_THROW((void)source.createExclusive(name()),std::logic_error);
    EXPECT_THROW((void)source.openExisting(name(),{},0),std::logic_error);
    auto first=destination.createExclusive(name());const int descriptor=first.descriptor();auto second=std::move(first);
    EXPECT_EQ(first.descriptor(),-1);
    EXPECT_EQ(second.descriptor(),descriptor);
    EXPECT_THROW(first.unlink(),std::logic_error);
    EXPECT_THROW(first.requireExtent(0),std::logic_error);
}
TEST(MPIOwnedSharedBacking, UnownedExternalProcessDoesNotAdoptInheritedPaths)
{
    const auto child=::fork();
    ASSERT_GE(child,0);
    if(child==0) {
        for(int descriptor=5;descriptor<=8;++descriptor)::close(descriptor);
        ::unsetenv("OMPI_MCA_llaminar_session_protocol");::unsetenv("OMPI_MCA_llaminar_session_ready");
        ::setenv("OMPI_COMM_WORLD_SIZE","2",1);
        try {::_exit(MPISessionOwner::sharedDirectory()?5:0);}catch(...) {::_exit(6);}
    }
    int status=0;
    ASSERT_EQ(::waitpid(child,&status,0),child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status),0);
}
/** @brief Exercise the original inherited loans without changing any runtime files. */
TEST(MPIOwnedSharedBacking, GuardianRetiresAfterChildEnumeratesInheritedDirectoryLoans)
{
    (void)directory(); // Validate the real immutable startup witness first.
    for (const int original : {6, 7}) {
        const int duplicate=::dup(original);
        ASSERT_GE(duplicate,0);
        DIR *entries=::fdopendir(duplicate);
        if(!entries) {::close(duplicate);FAIL()<<"Cannot inspect original startup loan";}
        unsigned count=0;
        errno=0;
        while(::readdir(entries))++count;
        const int error=errno;
        EXPECT_EQ(::closedir(entries),0);
        EXPECT_EQ(error,0);
        EXPECT_GE(count,3u); // '.', '..', and the guardian's immutable owner marker.
    }
    // CTest additionally requires the real guardian's terminal status. Its
    // cleanup must start an independent enumeration even though both original
    // open-file-description cursors now point at EOF.
}
/** @brief Enter the actual process owner before running device-free assertions. */
int main(int argc,char **argv)
{
    if(auto terminal=MPISessionOwner::enterProcess(argc,argv))return *terminal;
    ::testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();
}
