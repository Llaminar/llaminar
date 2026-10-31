/**
 * @file Test__MoEOverlaySharedBackingStartup.cpp
 * @brief Fast actual startup decoder/file replay through an explicit CPU MPI link.
 *
 * No MPI runtime or accelerator is initialized. The real creator, exact file
 * authentication and three-phase state machine run against retained mock
 * packets; real native MPI fault/exit regressions complement these tests.
 */
#include "execution/moe/MoEOverlaySharedBackingStartup.h"
#include "fixtures/mpi/OwnedSharedBackingReplay.h"
#include "mocks/MockMPITopology.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @brief Two explicitly modeled MPI endpoints sharing the actual owner's file namespace. */
struct Pair final {
    std::shared_ptr<OwnedSharedBackingReplayLink> link=std::make_shared<OwnedSharedBackingReplayLink>();
    std::array<std::shared_ptr<OwnedSharedBackingReplayContext>,2> contexts;
    std::array<MPISessionOwner::SharedFile,2> files;
    std::array<std::exception_ptr,2> errors;
    const std::array<int,2> participants{0,1};
    std::string basename;
    MoEOverlaySharedBackingEpoch epoch;
    Pair():epoch{MoEOverlaySharedBackingKind::ActivationRows,0,17,23,29}
    {
        static std::atomic<unsigned> sequence{0};
        basename="startup-replay-"+std::to_string(::getpid())+"-"+std::to_string(++sequence);
        for(int rank=0;rank<2;++rank) {
            contexts[rank]=std::make_shared<OwnedSharedBackingReplayContext>(rank,2);
            contexts[rank]->set_topology(MockMPITopology::createSimple(rank,2,2));
            contexts[rank]->bind(link);
        }
        epoch.node_namespace=contexts[0]->topology()->node_shared_memory_namespace();
    }
    /**
     * @brief Execute the same production setup entry on both endpoint threads.
     * @param bytes Exact physical extent authenticated independently by both ranks.
     */
    void execute(std::uint64_t bytes=4096)
    {
        std::array<std::thread,2> workers;
        for(int rank=0;rank<2;++rank)workers[rank]=std::thread([&,rank]{
            try {files[rank]=establishMoEOverlaySharedBacking(*contexts[rank],epoch,0,participants,basename,bytes);}
            catch(...) {errors[rank]=std::current_exception();}
        });
        for(auto &worker:workers)worker.join();
    }
};
/** @brief Decode only the actual canonical declaration; never create substitute producer facts. */
std::optional<detail::MoEOverlaySharedBackingDeclarationV1> declaration(const OwnedSharedBackingReplayMessage &message)
{
    if(message.source!=0 || message.bytes.size()!=sizeof(detail::MoEOverlaySharedBackingDeclarationV1))return std::nullopt;
    detail::MoEOverlaySharedBackingDeclarationV1 value;std::memcpy(&value,message.bytes.data(),sizeof(value));return value;
}
}
TEST(MoEOverlaySharedBackingStartup, ActualHandshakeReturnsOneOriginalFileWithOnlyLiveMetadata)
{
    Pair pair;std::vector<std::size_t> source_bytes,peer_bytes;
    pair.link->before_send=[&](const auto &message){std::lock_guard lock(pair.link->mutex);
        (message.source==0?source_bytes:peer_bytes).push_back(message.bytes.size());};
    pair.execute();
    EXPECT_FALSE(pair.errors[0]);
    EXPECT_FALSE(pair.errors[1]);
    EXPECT_EQ(pair.files[0].identity(),pair.files[1].identity());
    EXPECT_EQ(source_bytes,(std::vector<std::size_t>{sizeof(detail::MoEOverlaySharedBackingDeclarationV1),pair.basename.size(),sizeof(detail::MoEOverlaySharedBackingReplyV1)}));
    EXPECT_EQ(peer_bytes,(std::vector<std::size_t>{sizeof(detail::MoEOverlaySharedBackingReplyV1)}));
    EXPECT_TRUE(pair.link->messages.empty());
    pair.files[0].unlink();
    EXPECT_FALSE(std::filesystem::exists(pair.files[0].path()));
    EXPECT_NO_THROW(pair.files[1].requireExtent(4096));
}
TEST(MoEOverlaySharedBackingStartup, CreatorFailureDrainsAbortWithoutRemovingPreexistingNeighbor)
{
    Pair pair;auto directory=MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(directory);
    auto neighbor=directory->createExclusive(pair.basename);
    ASSERT_EQ(::ftruncate(neighbor.descriptor(),4096),0);
    const auto original=neighbor.identity();pair.execute();
    EXPECT_TRUE(pair.errors[0]);
    EXPECT_TRUE(pair.errors[1]);
    EXPECT_EQ(pair.files[0].descriptor(),-1);
    EXPECT_EQ(pair.files[1].descriptor(),-1);
    EXPECT_EQ(neighbor.identity(),original);
    EXPECT_NO_THROW(neighbor.requireExtent(4096));
    EXPECT_TRUE(std::filesystem::exists(neighbor.path()));
    EXPECT_TRUE(pair.link->messages.empty());
}
TEST(MoEOverlaySharedBackingStartup, PeerRejectsChangedPhysicalExtentBeforeAnyCommitOrMapping)
{
    Pair pair;auto directory=MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(directory);bool mutated=false;
    pair.link->before_send=[&](const auto &message){if(auto original=declaration(message)) {
        EXPECT_EQ(original->bytes,4096u);
    ASSERT_EQ(::truncate((directory->path()+"/"+pair.basename).c_str(),4095),0);mutated=true;
    }};
    pair.execute();
    EXPECT_TRUE(mutated);
    EXPECT_TRUE(pair.errors[0]);
    EXPECT_TRUE(pair.errors[1]);
    EXPECT_EQ(pair.files[0].descriptor(),-1);
    EXPECT_EQ(pair.files[1].descriptor(),-1);
    EXPECT_TRUE(pair.link->messages.empty());
    EXPECT_FALSE(std::filesystem::exists(directory->path()+"/"+pair.basename));
}
TEST(MoEOverlaySharedBackingStartup, PeerRejectsReplacementInodeAndCreatorCannotUnlinkReplacement)
{
    Pair pair;auto directory=MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(directory);
    MPISessionOwner::SharedFile replacement;std::string saved;MPISessionOwner::SharedIdentity original{};
    pair.link->before_send=[&](const auto &message){if(auto created=declaration(message)) {
        original=created->file;const auto path=directory->path()+"/"+pair.basename;saved=path+"-original";
        ASSERT_EQ(::rename(path.c_str(),saved.c_str()),0);replacement=directory->createExclusive(pair.basename);
        ASSERT_EQ(::ftruncate(replacement.descriptor(),4096),0);
    }};
    pair.execute();
    EXPECT_TRUE(pair.errors[0]);
    EXPECT_TRUE(pair.errors[1]);
    EXPECT_TRUE(pair.link->messages.empty());
    ASSERT_GE(replacement.descriptor(),0);
    EXPECT_NE(replacement.identity(),original);
    EXPECT_TRUE(std::filesystem::exists(replacement.path()));replacement.unlink();
    struct stat before{};
    ASSERT_EQ(::lstat(saved.c_str(),&before),0);
    EXPECT_EQ(before.st_ino,original.inode);
    EXPECT_EQ(::unlink(saved.c_str()),0);
}

/** @test The original no-op MPI fixture cannot certify a startup ACK. */
TEST(MoEOverlaySharedBackingStartup, PlainMockOriginalControlRejectsMissingAckBytes)
{
    Pair pair;
    MockMPIContext original(0,2);
    original.set_topology(MockMPITopology::createSimple(0,2,2));
    EXPECT_EXIT((void)establishMoEOverlaySharedBacking(original,pair.epoch,0,
        pair.participants,pair.basename,4096), ::testing::ExitedWithCode(1),
        "startup ACK has partial/wrong live extent");
}
/** @test Both modeled ranks replay changed epochs and live extents through the actual decoder. */
TEST(MoEOverlaySharedBackingStartup, BoundPairReplaysFreshEpochsAndLargeToSmallFiles)
{
    Pair pair;
    for(unsigned replay=0;replay<3;++replay) {
        pair.epoch.generation+=1;
        const std::uint64_t bytes=replay==0?8192:(replay==1?4096:128);
        pair.execute(bytes);
        ASSERT_FALSE(pair.errors[0]);ASSERT_FALSE(pair.errors[1]);
        EXPECT_EQ(pair.files[0].identity(),pair.files[1].identity());
        EXPECT_NO_THROW(pair.files[0].requireExtent(bytes));
        EXPECT_NO_THROW(pair.files[1].requireExtent(bytes));
        EXPECT_TRUE(pair.link->messages.empty());
        pair.files[0].unlink();
        pair.files={};
    }
}
