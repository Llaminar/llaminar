/**
 * @file Test__TransferSpec.cpp
 * @brief Unit tests for TransferSpec transfer mechanism derivation
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for TransferSpec::derive() which determines how activation data
 * should be transferred between adjacent PP stages:
 * - LOCAL_PP: Same-rank transfer using NCCL/HOST/host-staged
 * - MPI_INTRAHOST: Cross-rank on same machine
 * - MPI_INTERHOST: Cross-rank on different machines
 *
 * All tests are pure data-structure tests — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>

#include "execution/parallelism_tree/TransferSpec.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;

// =============================================================================
// Test Suite: Transfer Mechanism Derivation
// =============================================================================

/**
 * @test SameRankLocalPP
 * @brief Two TP domains on same rank → LOCAL_PP
 *
 * When the "from" and "to" subtrees share at least one MPI rank,
 * the transfer is local (no MPI needed).
 */
TEST(Test__TransferSpec, SameRankLocalPP)
{
    // Two TP domains on rank 0
    // PP(root)
    // ├── TP(tp0, [cuda:0, cuda:1], rank=0)
    // └── TP(tp1, [cuda:2, cuda:3], rank=0)

    auto from = TP("tp0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0);
    auto to = TP("tp1", {GlobalDeviceAddress::cuda(2), GlobalDeviceAddress::cuda(3)}, 0);

    auto spec = TransferSpec::derive(from, to, 100);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::LOCAL_PP);
    EXPECT_EQ(spec.sender_rank, -1) << "LOCAL_PP should have sender_rank = -1";
    EXPECT_EQ(spec.receiver_rank, -1) << "LOCAL_PP should have receiver_rank = -1";
    EXPECT_EQ(spec.local_backend, CollectiveBackendType::AUTO);
    EXPECT_EQ(spec.mpi_tag, 100);
}

/**
 * @test SameRankSingleDevices
 * @brief Two single DEVICE nodes on same rank → LOCAL_PP
 */
TEST(Test__TransferSpec, SameRankSingleDevices)
{
    auto from = Device(GlobalDeviceAddress::cuda(0), 0);
    auto to = Device(GlobalDeviceAddress::cuda(1), 0);

    auto spec = TransferSpec::derive(from, to, 42);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::LOCAL_PP);
    EXPECT_EQ(spec.sender_rank, -1);
    EXPECT_EQ(spec.receiver_rank, -1);
    EXPECT_EQ(spec.mpi_tag, 42);
}

/**
 * @test CrossRankSameHost
 * @brief Two nodes on different ranks, same hostname → MPI_INTRAHOST
 *
 * Different MPI ranks but same machine (detected via hostname match).
 */
TEST(Test__TransferSpec, CrossRankSameHost)
{
    // Both on "localhost", different ranks
    auto from = Device(GlobalDeviceAddress::cuda(0, 0, "localhost"), 0);
    auto to = Device(GlobalDeviceAddress::cuda(0, 0, "localhost"), 1);

    auto spec = TransferSpec::derive(from, to, 200);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTRAHOST);
    EXPECT_EQ(spec.sender_rank, 0);
    EXPECT_EQ(spec.receiver_rank, 1);
    EXPECT_EQ(spec.mpi_tag, 200);
}

/**
 * @test CrossRankSameHostMultiDevice
 * @brief TP domains on different ranks, same hostname → MPI_INTRAHOST
 */
TEST(Test__TransferSpec, CrossRankSameHostMultiDevice)
{
    // TP on rank 0 with devices on "node0"
    auto from = TP("tp0", {GlobalDeviceAddress::cuda(0, 0, "node0"), GlobalDeviceAddress::cuda(1, 0, "node0")},
                   0);

    // TP on rank 1 with devices on "node0" (same host, different rank)
    auto to = TP("tp1", {GlobalDeviceAddress::cuda(2, 1, "node0"), GlobalDeviceAddress::cuda(3, 1, "node0")},
                 1);

    auto spec = TransferSpec::derive(from, to, 300);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTRAHOST);
    EXPECT_EQ(spec.sender_rank, 0);
    EXPECT_EQ(spec.receiver_rank, 1);
    EXPECT_EQ(spec.mpi_tag, 300);
}

/**
 * @test CrossRankDifferentHost
 * @brief Two nodes on different ranks, different hostnames → MPI_INTERHOST
 */
TEST(Test__TransferSpec, CrossRankDifferentHost)
{
    auto from = Device(GlobalDeviceAddress::cuda(0, 0, "node0"), 0);
    auto to = Device(GlobalDeviceAddress::cuda(0, 0, "node1"), 1);

    auto spec = TransferSpec::derive(from, to, 400);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTERHOST);
    EXPECT_EQ(spec.sender_rank, 0);
    EXPECT_EQ(spec.receiver_rank, 1);
    EXPECT_EQ(spec.mpi_tag, 400);
}

/**
 * @test CrossRankDifferentHostComplex
 * @brief Complex tree with multiple hosts → MPI_INTERHOST
 *
 * Tests a PP stage from host0 to host1and in a 4x2x2 topology.
 */
TEST(Test__TransferSpec, CrossRankDifferentHostComplex)
{
    // Build host0 PP stage: two TP domains on ranks 0 and 1
    auto from = PP("host0", {
                                TP("socket0", {GlobalDeviceAddress::cuda(0, 0, "host0"), GlobalDeviceAddress::cuda(1, 0, "host0")},
                                   0),
                                TP("socket1", {GlobalDeviceAddress::cuda(0, 1, "host0"), GlobalDeviceAddress::cuda(1, 1, "host0")},
                                   1),
                            });

    // Build host1 PP stage: two TP domains on ranks 2 and 3
    auto to = PP("host1", {
                              TP("socket2", {GlobalDeviceAddress::cuda(0, 0, "host1"), GlobalDeviceAddress::cuda(1, 0, "host1")},
                                 2),
                              TP("socket3", {GlobalDeviceAddress::cuda(0, 1, "host1"), GlobalDeviceAddress::cuda(1, 1, "host1")},
                                 3),
                          });

    auto spec = TransferSpec::derive(from, to, 500);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTERHOST);
    // sender_rank = max of from's leafRanks = 1
    // receiver_rank = min of to's leafRanks = 2
    EXPECT_EQ(spec.sender_rank, 1);
    EXPECT_EQ(spec.receiver_rank, 2);
    EXPECT_EQ(spec.mpi_tag, 500);
}

/**
 * @test TagAssignment
 * @brief Verify tag_base is used correctly
 */
TEST(Test__TransferSpec, TagAssignment)
{
    auto from = Device(GlobalDeviceAddress::cuda(0), 0);
    auto to = Device(GlobalDeviceAddress::cuda(1), 0);

    // Test various tag values
    for (int tag : {0, 1, 100, 65535, 1000000})
    {
        auto spec = TransferSpec::derive(from, to, tag);
        EXPECT_EQ(spec.mpi_tag, tag) << "Tag " << tag << " not preserved";
    }
}

/**
 * @test MechanismName
 * @brief Verify mechanismName() static function
 */
TEST(Test__TransferSpec, MechanismName)
{
    EXPECT_STREQ(TransferSpec::mechanismName(TransferSpec::Mechanism::LOCAL_PP), "LOCAL_PP");
    EXPECT_STREQ(TransferSpec::mechanismName(TransferSpec::Mechanism::MPI_INTRAHOST), "MPI_INTRAHOST");
    EXPECT_STREQ(TransferSpec::mechanismName(TransferSpec::Mechanism::MPI_INTERHOST), "MPI_INTERHOST");
}

/**
 * @test ToString
 * @brief Verify human-readable output
 */
TEST(Test__TransferSpec, ToString)
{
    // LOCAL_PP case
    auto from = Device(GlobalDeviceAddress::cuda(0), 0);
    auto to = Device(GlobalDeviceAddress::cuda(1), 0);
    auto spec1 = TransferSpec::derive(from, to, 100);

    std::string str1 = spec1.toString();
    EXPECT_NE(str1.find("LOCAL_PP"), std::string::npos);
    EXPECT_NE(str1.find("local_backend"), std::string::npos);

    // MPI_INTERHOST case
    auto from2 = Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 0);
    auto to2 = Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 1);
    auto spec2 = TransferSpec::derive(from2, to2, 200);

    std::string str2 = spec2.toString();
    EXPECT_NE(str2.find("MPI_INTERHOST"), std::string::npos);
    EXPECT_NE(str2.find("sender_rank: 0"), std::string::npos);
    EXPECT_NE(str2.find("receiver_rank: 1"), std::string::npos);
    EXPECT_NE(str2.find("mpi_tag: 200"), std::string::npos);
}

/**
 * @test Equality
 * @brief Verify equality operator
 */
TEST(Test__TransferSpec, Equality)
{
    auto from = Device(GlobalDeviceAddress::cuda(0), 0);
    auto to = Device(GlobalDeviceAddress::cuda(1), 0);

    auto spec1 = TransferSpec::derive(from, to, 100);
    auto spec2 = TransferSpec::derive(from, to, 100);

    EXPECT_EQ(spec1, spec2);

    // Different tag
    auto spec3 = TransferSpec::derive(from, to, 200);
    EXPECT_NE(spec1, spec3);

    // Different mechanism
    auto from4 = Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 0);
    auto to4 = Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 1);
    auto spec4 = TransferSpec::derive(from4, to4, 100);
    EXPECT_NE(spec1, spec4);
}

/**
 * @test MultiRankFromSubtree
 * @brief Test sender_rank selection when from has multiple ranks
 *
 * sender_rank should be max of from's leafRanks.
 */
TEST(Test__TransferSpec, MultiRankFromSubtree)
{
    // from subtree spans ranks 0, 1, 2
    auto from = PP("from", {
                               Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 0),
                               Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 1),
                               Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 2),
                           });

    // to is on rank 3
    auto to = Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 3);

    auto spec = TransferSpec::derive(from, to, 600);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTERHOST);
    EXPECT_EQ(spec.sender_rank, 2) << "Should be max of {0,1,2} = 2";
    EXPECT_EQ(spec.receiver_rank, 3);
}

/**
 * @test MultiRankToSubtree
 * @brief Test receiver_rank selection when to has multiple ranks
 *
 * receiver_rank should be min of to's leafRanks.
 */
TEST(Test__TransferSpec, MultiRankToSubtree)
{
    // from is on rank 0
    auto from = Device(GlobalDeviceAddress::cuda(0, 0, "host0"), 0);

    // to subtree spans ranks 2, 3, 4
    auto to = PP("to", {
                           Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 2),
                           Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 3),
                           Device(GlobalDeviceAddress::cuda(0, 0, "host1"), 4),
                       });

    auto spec = TransferSpec::derive(from, to, 700);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::MPI_INTERHOST);
    EXPECT_EQ(spec.sender_rank, 0);
    EXPECT_EQ(spec.receiver_rank, 2) << "Should be min of {2,3,4} = 2";
}

/**
 * @test MixedVendorLocalPP
 * @brief Mixed CUDA+ROCm on same rank is still LOCAL_PP
 */
TEST(Test__TransferSpec, MixedVendorLocalPP)
{
    auto from = TP("tp0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)}, 0);
    auto to = TP("tp1", {GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::rocm(1)}, 0);

    auto spec = TransferSpec::derive(from, to, 800);

    EXPECT_EQ(spec.mechanism, TransferSpec::Mechanism::LOCAL_PP);
    EXPECT_EQ(spec.sender_rank, -1);
    EXPECT_EQ(spec.receiver_rank, -1);
}
